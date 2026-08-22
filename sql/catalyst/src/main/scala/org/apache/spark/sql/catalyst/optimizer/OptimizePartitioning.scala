/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.catalyst.optimizer

import scala.collection.mutable

import org.apache.spark.SparkException
import org.apache.spark.sql.catalyst.expressions.{AliasHelper, Attribute, AttributeMap, Expression}
import org.apache.spark.sql.catalyst.planning.{ExtractEquiJoinKeys, NodeWithDistributionRequirementsAboveReusedSubplans, NodeWithProjectFilterAboveReusedSubplan}
import org.apache.spark.sql.catalyst.plans.QueryPlan
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreePattern.REPARTITION_OPERATION
import org.apache.spark.sql.internal.SQLConf

/**
 * Chooses the partitioning of the shuffle that backs a reused subplan if all consumers of the
 * reused subplans have the same distribution requirements, so consumers do not have to add a
 * redundant shuffle of their own on top of it.
 *
 * A repartition is upgraded to a [[RepartitionByExpression]] only when EVERY consumer of the
 * reused subplan (every reference sharing that repartition id) has a distribution requirement we
 * could detect, and all of them agree on the same partitioning. There are 3 cases:
 *
 * 1. All consumers have the same detected distribution requirement. Replace the repartition with a
 * RepartitionByExpression on those keys.
 * E.g:  Union
 *         Join(leftKeys = [a,b,c], rightKeys = [d,e,f])
 *           Repartition (id = 1)
 *             Subplan
 *           Table
 *         HashAggregate(groupBy = [a,b,c])
 *           Repartition (id = 1)
 *             Subplan
 *
 * 2. Consumers have different detected distribution requirements. This includes the situation that
 * consumers might have partition expressions in common. But we don't generate a
 * RepartitionByExpression based on common keys here, since it might lead to low parallelism /
 * skewed shuffles which can be the performance bottleneck. We fall back to Repartition which is
 * converted to a shuffle with LocalPartitioning later.
 * E.g:  Union
 *         Join(leftKeys = [a,b,c], rightKeys = [d,e,f])
 *           Repartition (id = 1)
 *             Subplan
 *           Table
 *         HashAggregate (groupBy = [b, d])
 *           Repartition (id = 1)
 *             Subplan
 *
 * 3. At least one consumer has no detected distribution requirement. We deliberately fall back to
 * LocalPartition rather than partitioning on the requirements we did find. A missing requirement
 * is not necessarily a consumer that truly needs none: it can also be a requirement that exists
 * above the reuse but that our patterns failed to extract (e.g. an aggregate over an Expand), or a
 * failure to propagate aliases down to the subplan. Committing to a HashPartitioning that serves
 * only the consumers we could analyze risks a skewed / low-parallelism shuffle for the ones we
 * could not, so we only force a partitioning when every consumer votes for it.
 * E.g:  Union
 *         Join(leftKeys = [a,b,c], rightKeys = [d,e,f])
 *           Repartition (id = 1)
 *             Subplan
 *           Table
 *         Repartition (id = 1)
 *           Subplan
 *
 * The `LocalPartition` fallback of cases 2 and 3 is only safe when every reference is a plain
 * [[Repartition]]. If a reference is already a [[RepartitionByExpression]] (which states a
 * mandatory partitioning), we can neither downgrade the shared shuffle to `LocalPartition` without
 * breaking that consumer nor pick a shared clustering when there is no consensus. That state is
 * unreachable under correct repartition generation, so we raise an internal error rather than
 * mis-partition.
 *
 * This runs between [[ReplaceCTERefWithRepartition]] (which turns CTE references into plan-reuse
 * repartitions) and [[ReplaceRepartitionWithCTEReuse]] (which turns those into CTEReuseRelation).
 * It is an optional, performance-only rule: it is gated on
 * [[SQLConf.REPLACE_CTE_REF_WITH_CTE_REUSE]] and can be disabled independently
 * without affecting correctness.
 *
 * Like [[ReplaceRepartitionWithCTEReuse]], it runs only on the main query plan; a subquery
 * optimized on its own does not expose the full set of references it needs to see across the plan.
 */
object OptimizePartitioning extends Rule[LogicalPlan] with AliasHelper {
  def apply(plan: LogicalPlan): LogicalPlan = {
    // Skip a subquery optimized on its own (root is a Subquery); only the main query pass sees
    // every reference of a reused subplan. See the class doc.
    if (plan.isInstanceOf[Subquery] ||
        !conf.getConf(SQLConf.REPLACE_CTE_REF_WITH_CTE_REUSE) ||
        !plan.containsPattern(REPARTITION_OPERATION)) {
      return plan
    }

    val requirements = mutable.HashMap.empty[Long, RequirementsState]
    val detected = new java.util.IdentityHashMap[PlanReusableRepartition, Seq[Expression]]()

    plan.foreachWithSubqueriesAndPruning(_.containsPattern(REPARTITION_OPERATION)) {
      case r: PlanReusableRepartition if r.isForPlanReuse =>
        val state = requirements.getOrElseUpdate(r.repartitionId, new RequirementsState)
        state.referenceCount += 1
        r match {
          case rbe: RepartitionByExpression if rbe.partitionExpressions.nonEmpty =>
            state.containsClusteringRepartition = true
          case _ =>
        }
      case _ =>
    }

    plan.foreachWithSubqueriesAndPruning(_.containsPattern(REPARTITION_OPERATION)) {
      case NodeWithDistributionRequirementsAboveReusedSubplans(found) =>
        found.foreach {
          case (Some(requirement), repartition) if !(requirement eq repartition) =>
            requiredClustering(requirement, repartition).foreach(detected.put(repartition, _))
          case (requirementOpt, repartition) =>
            Option(detected.get(repartition))
              .orElse(requirementOpt.flatMap(requiredClustering(_, repartition)))
              .foreach(requirements(repartition.repartitionId).partitionExpressions += _)
        }
      case _ =>
    }

    val chosen = requirements.flatMap { case (id, state) =>
      val res = agreedClustering(id, state.partitionExpressions.toSeq, state.referenceCount)
      // If the subplan has a reference with RepartitionByExpression above it, but cannot
      // find an agreedClustering, then throw errors. This means the logic to generate
      // repartitions or subplan candidates to be reused is wrong.
      if (res.isEmpty && state.containsClusteringRepartition) {
        throw SparkException.internalError(
          s"Plan-reuse repartition id=$id has a RepartitionByExpression consumer, but its " +
            s"consumers do not agree on a partitioning; the shared shuffle cannot serve all of " +
            s"them. This indicates a repartition-generation bug.")
      }
      res.map(id -> _)
    }.toMap
    if (chosen.isEmpty) plan else applyPartitionings(plan, chosen)
  }

  /**
   * Return the distribution requirement related expressions using the subplan's attributes, or
   * None if the subplan's shuffle cannot serve it.
   *
   * The distribution requirement is collected from: 1. aggregate 2. window 3. equi-join which is
   * expected to be planned as SHJ/SMJ. The input `requirement` and `repartition` is selected from
   * [[NodeWithDistributionRequirementsAboveReusedSubplans]].
   */
  private def requiredClustering(
      requirement: LogicalPlan,
      repartition: PlanReusableRepartition): Option[Seq[Expression]] = requirement match {
    case rbe: RepartitionByExpression if rbe eq repartition =>
      // The node already states its own clustering. Its keys must be expressible over its child,
      // else the shuffle cannot partition by them.
      assert(
        rbe.partitionExpressions.forall(_.references.subsetOf(rbe.child.outputSet)),
        s"Plan-reuse RepartitionByExpression partitions by expressions that its child does not " +
          s"output. Child output: ${rbe.child.output.mkString("[", ", ", "]")}\n$rbe")
      Some(normalizedClustering(rbe.partitionExpressions, repartition.output))
    case _ =>
      val keys = requirement match {
        case agg: Aggregation => agg.groupingExpressions
        case w: Window => w.partitionSpec
        case ExtractEquiJoinKeys(_, leftKeys, rightKeys, _, _, left, right, _) =>
          if (isAboveSubplan(left, repartition)) leftKeys else rightKeys
        case other =>
          // Defensive throws. This branch should not be triggered since all requirement
          // plan is collected from the above patterns.
          throw SparkException.internalError(
            s"Unexpected distribution requirement operator: ${other.nodeName}")
      }
      val child = requirement.children.find(isAboveSubplan(_, repartition)).get
      val mapped = substituteAliasesDownTo(child, keys)
      // A key still referencing something outside the subplan cannot be served by its shuffle.
      if (mapped.forall(_.references.subsetOf(repartition.outputSet))) {
        // NormalizeClustering so the attribute references from mapped expressions are rewritten
        // based on the ordinal in repartition.output.
        Some(normalizedClustering(mapped, repartition.output))
      } else {
        None
      }
  }

  /**
   * Rewrites `exprs` into the attributes of the reused subplan below `plan`, substituting each
   * node's alias map on the way down. Must cross the same nodes as
   * [[NodeWithProjectFilterAboveReusedSubplan]].
   */
  private def substituteAliasesDownTo(
      plan: LogicalPlan,
      exprs: Seq[Expression]): Seq[Expression] = plan match {
    case p: Project => substituteAliasesDownTo(p.child, exprs.map(replaceAlias(_, getAliasMap(p))))
    case f: Filter => substituteAliasesDownTo(f.child, exprs)
    case a: AggregatePart if !a.isFinalAggregate =>
      substituteAliasesDownTo(a.child, exprs.map(replaceAlias(_, getAliasMap(a))))
    case s: SecureViewUnaryNode => substituteAliasesDownTo(s.child, exprs)
    case _ => exprs
  }

  /** Whether `repartition` is the reused subplan reached from `plan`. */
  private def isAboveSubplan(
      plan: LogicalPlan,
      repartition: PlanReusableRepartition): Boolean = plan match {
    case NodeWithProjectFilterAboveReusedSubplan(r) => r eq repartition
    case _ => false
  }

  /**
   * Rewrites each attribute's exprId to its ordinal in `output` so requirements stated against
   * different references to one subplan compare equal.
   */
  private def normalizedClustering(
      keys: Seq[Expression],
      output: Seq[Attribute]): Seq[Expression] = {
    keys.map(QueryPlan.normalizeExpressions(_, output))
  }

  /**
   * Return the agreeed clustering using the attributes in the required.head if all requirements
   * for the same repartition id has semantically same distribution requirements and requirements
   * size equals to reference count. Otherwise, return None.
   */
  private def agreedClustering(
      repartitionId: Long,
      required: Seq[Seq[Expression]],
      referenceCount: Int): Option[Seq[Expression]] = {
    if (required.isEmpty || required.size != referenceCount) {
      logDebug(s"CTE reuse partitioning for repartitionId=$repartitionId keeps LocalPartition: " +
        s"only ${required.size} of $referenceCount consumers have a detected clustering " +
        s"requirement")
      None
    } else if (!required.tail.forall(sameKeys(_, required.head))) {
      logDebug(s"CTE reuse partitioning for repartitionId=$repartitionId keeps LocalPartition: " +
        s"requirements disagree ($required)")
      None
    } else {
      logDebug(s"CTE reuse partitioning for repartitionId=$repartitionId is ${required.head}")
      Some(required.head)
    }
  }

  /**
   * Whether two clusterings are the same partitioning. This is order-sensitive on purpose: we
   * deliberately do NOT treat (A, B) and (B, A) as the same, because HashPartitioning(A, B) is not
   * the same partitioning as HashPartitioning(B, A) -- they route rows to different partitions, and
   * `ClusteredDistribution.areAllClusterKeysMatched` (used when a consumer sets
   * `requireAllClusterKeys`) is itself order-sensitive. Treating them as equal could pick a
   * partitioning that such a consumer would not accept.
   */
  private def sameKeys(left: Seq[Expression], right: Seq[Expression]): Boolean = {
    left.length == right.length && left.zip(right).forall {
      case (l, r) => l.semanticEquals(r)
    }
  }

  /**
   * Replaces each eligible plan-reuse [[Repartition]] with a [[RepartitionByExpression]] on the
   * chosen clustering, keeping `repartitionId` and `repartitionOrigin` so reuse keeps working.
   */
  private def applyPartitionings(
      plan: LogicalPlan,
      chosen: Map[Long, Seq[Expression]]): LogicalPlan = {
    plan.transformUpWithSubqueriesAndPruning(_.containsPattern(REPARTITION_OPERATION)) {
      case r: Repartition if r.isForPlanReuse && chosen.contains(r.repartitionId) =>
        val keys = denormalizedClustering(chosen(r.repartitionId), r.output)
        RepartitionByExpression(
          partitionExpressions = keys,
          child = r.child,
          optNumPartitions = None,
          id = r.repartitionId,
          repartitionOrigin = r.repartitionOrigin)
    }
  }

  /**
   * The reverse of [[normalizedClustering]]: maps ordinal-keyed attributes back onto one
   * reference's own, since foreign exprIds would leave the plan unresolvable.
   */
  private def denormalizedClustering(
      keys: Seq[Expression],
      output: Seq[Attribute]): Seq[Expression] = {
    val reverseMapping = AttributeMap(output.map { a =>
      QueryPlan.normalizeExpressions(a, output) -> a
    })
    keys.map(_.transformUp {
      case a: Attribute => reverseMapping.getOrElse(a, a)
    })
  }

  /**
   * Accumulated state for one plan-reuse repartition id while choosing its partitioning.
   *
   * @param referenceCount               number of references (repartition nodes) sharing this id.
   * @param containsClusteringRepartition whether any reference is a [[RepartitionByExpression]]
   *                                     with partition expressions (a Hash/Range clustering) that a
   *                                     `LocalPartition` fallback cannot honor. An expression-less
   *                                     RepartitionByExpression (RoundRobin/Single) does not count.
   * @param partitionExpressions         the clustering detected for each reference, one entry per
   *                                     reference for which a requirement was found (references
   *                                     with no detectable requirement contribute nothing).
   */
  private class RequirementsState {
    private[OptimizePartitioning] var referenceCount: Int = 0
    private[OptimizePartitioning] var containsClusteringRepartition: Boolean = false
    private[OptimizePartitioning] val partitionExpressions: mutable.ArrayBuffer[Seq[Expression]] =
      mutable.ArrayBuffer.empty
  }
}

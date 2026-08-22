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

package org.apache.spark.sql

import scala.collection.mutable



import org.apache.spark.sql.catalyst.plans.logical.CTEReuseRelation
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, LocalPartition}
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.execution.exchange.{ENSURE_REQUIREMENTS, ShuffleExchangeLike}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession

/**
 * Tests for [[org.apache.spark.sql.catalyst.optimizer.OptimizePartitioning]]: the
 * partitioning chosen for a reused subplan's shuffle, and the redundant consumer shuffles that
 * choice removes.
 */
class CTEReusePartitioningSuite extends QueryTest with SharedSparkSession
    with AdaptiveSparkPlanHelper {

  private def withCTEReuse(f: => Unit): Unit = {
    withSQLConf(
      "spark.sql.optimizer.replaceCTERefWithCTEReuse.enabled" -> "true",
      "spark.sql.optimizer.useLocalShuffleForCTEReuse.enabled" -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      // Force shuffle joins: a broadcast join needs no clustered distribution, so its side
      // abstains from the partitioning vote.
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "-1"
    )(f)
  }

  private def withSrc(f: => Unit): Unit = {
    withTable("part_src", "part_other") {
      sql("CREATE TABLE part_src (a INT, b INT, c INT, v INT) USING parquet")
      sql(
        """INSERT INTO part_src VALUES
          |(1, 1, 1, 10), (2, 2, 2, 20), (3, 3, 3, 30),
          |(4, 4, 4, 40), (5, 5, 5, 50)""".stripMargin)
      sql("CREATE TABLE part_other (a INT, w INT) USING parquet")
      sql("INSERT INTO part_other VALUES (1, 100), (2, 200), (3, 300)")
      f
    }
  }

  private def cteReuseRelations(df: DataFrame): Seq[CTEReuseRelation] =
    df.queryExecution.optimizedPlan.collectWithSubqueries { case r: CTEReuseRelation => r }

  /** The partitioning every CTEReuseRelation reports; asserts they all agree. */
  private def cteePartitioning(df: DataFrame): Seq[String] = {
    val relations = cteReuseRelations(df)
    assert(relations.nonEmpty,
      "Expected at least one CTEReuseRelation.\n" +
        df.queryExecution.optimizedPlan.treeString)
    relations.map(_.partitioning.getClass.getSimpleName).distinct
  }

  /**
   * Distinct physical shuffles in the final executed plan.
   *
   * Deduplicated by object identity: every reference to a reused subplan wraps the SAME shuffle
   * (all `CTEReuseQueryStageExec`s of one cteId share one inner AQE), and `collectWithSubqueries`
   * descends into each of them, so a plain count reports one shuffle per reference.
   */
  private def distinctShuffles(df: DataFrame): Seq[ShuffleExchangeLike] = {
    df.collect()
    val found = collectWithSubqueries(df.queryExecution.executedPlan) {
      case s: ShuffleExchangeLike => s
    }
    val seen = mutable.LinkedHashMap.empty[Int, ShuffleExchangeLike]
    found.foreach(s => seen.getOrElseUpdate(System.identityHashCode(s), s))
    seen.values.toSeq
  }

  /** Number of distinct ENSURE_REQUIREMENTS shuffles in the final executed plan. */
  private def ensureRequirementsShuffles(df: DataFrame): Int =
    distinctShuffles(df).count(_.shuffleOrigin == ENSURE_REQUIREMENTS)

  // ---------------------------------------------------------------------------
  // Consensus -> HashPartitioning, and the redundant shuffles it removes
  // ---------------------------------------------------------------------------

  test("uniform join consumers force HashPartitioning and add no shuffle") {
    withCTEReuse {
      withSrc {
        // Both consumers are shuffle joins on the same keys in the same order, so the reuse
        // shuffle can produce that partitioning once and serve both.
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a FROM cte c1 JOIN cte c2 ON c1.a = c2.a AND c1.b = c2.b
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(classOf[HashPartitioning].getSimpleName),
          s"Expected HashPartitioning, got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        // The join's requirement is already satisfied by the reuse shuffle.
        assert(ensureRequirementsShuffles(df) == 0,
          "Expected no ENSURE_REQUIREMENTS shuffle above the CTE refs.\n" +
            df.queryExecution.executedPlan.treeString)
        checkAnswer(df, Seq(Row(1), Row(2), Row(3), Row(4), Row(5)))
      }
    }
  }

  test("Project and Filter between the CTE and its consumer do not block satisfaction") {
    withCTEReuse {
      withSrc {
        // Partitioning must survive a Project (which renames the keys) and a Filter. The physical
        // counterpart is ProjectExec being partitioning-preserving and FilterExec forwarding its
        // child's partitioning.
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT l.ka FROM
            |  (SELECT a as ka, v FROM cte WHERE v > 5) l
            |  JOIN
            |  (SELECT a as ka2, v FROM cte WHERE v > 15) rr
            |  ON l.ka = rr.ka2
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(classOf[HashPartitioning].getSimpleName),
          s"Expected HashPartitioning through Project/Filter, got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        assert(ensureRequirementsShuffles(df) == 0,
          "Expected no ENSURE_REQUIREMENTS shuffle across Project/Filter.\n" +
            df.queryExecution.executedPlan.treeString)
        checkAnswer(df, Seq(Row(2), Row(3), Row(4), Row(5)))
      }
    }
  }

  test("aggregate consumers on the same grouping keys force HashPartitioning") {
    withCTEReuse {
      withSrc {
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT a, sum(v) as s FROM cte GROUP BY a
            |UNION ALL
            |SELECT a, max(v) as s FROM cte GROUP BY a
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(classOf[HashPartitioning].getSimpleName),
          s"Expected HashPartitioning from aggregate consumers, got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        checkAnswer(df,
          Seq(Row(1, 10), Row(2, 20), Row(3, 30), Row(4, 40), Row(5, 50),
            Row(1, 10), Row(2, 20), Row(3, 30), Row(4, 40), Row(5, 50)))
      }
    }
  }

  // ---------------------------------------------------------------------------
  // No consensus -> LocalPartition
  // ---------------------------------------------------------------------------

  test("consumers requiring different keys fall back to LocalPartition") {
    withCTEReuse {
      withSrc {
        // One consumer clusters on `a`, the other on `b`. Partitioning on the common subset is
        // deliberately not attempted, so the fallback applies.
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a FROM cte c1 JOIN cte c2 ON c1.a = c2.a
            |UNION ALL
            |SELECT c3.b FROM cte c3 JOIN cte c4 ON c3.b = c4.b
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(LocalPartition.getClass.getSimpleName),
          s"Expected LocalPartition when consumers disagree, got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        df.collect()
      }
    }
  }

  test("key order mismatch is not consensus") {
    withCTEReuse {
      withSrc {
        // A join on (a, b) and an aggregate grouping by (b, a) require the same key SET but in a
        // different order. `requireAllClusterKeys` (which aggregates set) makes HashPartitioning
        // satisfy a distribution only for an exact ordered match, so these do not agree.
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a as x, c1.b as y FROM cte c1 JOIN cte c2
            |  ON c1.a = c2.a AND c1.b = c2.b
            |UNION ALL
            |SELECT b as x, a as y FROM cte GROUP BY b, a
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(LocalPartition.getClass.getSimpleName),
          s"Expected LocalPartition for a key-order mismatch, got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        df.collect()
      }
    }
  }

  test("broadcast join consumers abstain, leaving LocalPartition") {
    withSQLConf(
      "spark.sql.optimizer.replaceCTERefWithCTEReuse.enabled" -> "true",
      "spark.sql.optimizer.useLocalShuffleForCTEReuse.enabled" -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      // Large threshold: both joins are planned as broadcast, so neither requires a clustered
      // distribution and neither votes.
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "10485760"
    ) {
      withSrc {
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a FROM cte c1 JOIN part_other o1 ON c1.a = o1.a
            |UNION ALL
            |SELECT c2.a FROM cte c2 JOIN part_other o2 ON c2.a = o2.a
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(LocalPartition.getClass.getSimpleName),
          s"Expected LocalPartition when all consumers are broadcast joins, " +
            s"got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        df.collect()
      }
    }
  }

  test("a consumer with no requirement vetoes, falling back to LocalPartition") {
    withCTEReuse {
      withSrc {
        // One consumer is a plain scan of the CTE (no clustered distribution), the other is a
        // shuffle join. A consumer with no detected requirement vetoes: we only force a
        // partitioning when every consumer votes for it, so this falls back to LocalPartition.
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a FROM cte c1 JOIN cte c2 ON c1.a = c2.a
            |UNION ALL
            |SELECT a FROM cte
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(LocalPartition.getClass.getSimpleName),
          s"Expected LocalPartition (a no-requirement consumer vetoes), " +
            s"got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        df.collect()
      }
    }
  }

  test("computed join key reaches consensus") {
    withCTEReuse {
      withSrc {
        // `a + b` is not an output column of the CTE, but it is the same expression over the same
        // columns on both references, so it is a valid shared clustering.
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a FROM cte c1 JOIN cte c2 ON c1.a + c1.b = c2.a + c2.b
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(classOf[HashPartitioning].getSimpleName),
          s"Expected HashPartitioning for a computed key, got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        assert(ensureRequirementsShuffles(df) == 0,
          "Expected no ENSURE_REQUIREMENTS shuffle for a computed key.\n" +
            df.queryExecution.executedPlan.treeString)
        checkAnswer(df, Seq(Row(1), Row(2), Row(3), Row(4), Row(5)))
      }
    }
  }

  test("null-safe join key reaches consensus") {
    withCTEReuse {
      withSrc {
        // ExtractEquiJoinKeys rewrites `<=>` into two keys per side, Coalesce(a, default) and
        // IsNull(a) (see NullSafeEqualityExpressions). ShuffledJoin clusters on exactly those, so
        // partitioning the reuse shuffle by them satisfies the join.
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a FROM cte c1 JOIN cte c2 ON c1.a <=> c2.a
            |""".stripMargin)
        assert(cteePartitioning(df) == Seq(classOf[HashPartitioning].getSimpleName),
          s"Expected HashPartitioning for a null-safe key, got ${cteePartitioning(df)}.\n" +
            df.queryExecution.optimizedPlan.treeString)
        assert(ensureRequirementsShuffles(df) == 0,
          "Expected no ENSURE_REQUIREMENTS shuffle for a null-safe key.\n" +
            df.queryExecution.executedPlan.treeString)
        checkAnswer(df, Seq(Row(1), Row(2), Row(3), Row(4), Row(5)))
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Flag off
  // ---------------------------------------------------------------------------

  test("legacy path is unaffected when guaranteed reuse is disabled") {
    withSQLConf(
      "spark.sql.optimizer.replaceCTERefWithCTEReuse.enabled" -> "false",
      "spark.sql.optimizer.useLocalShuffleForCTEReuse.enabled" -> "true",
      SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> "true",
      SQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "-1"
    ) {
      withSrc {
        val df = sql(
          """WITH cte AS (SELECT a, b, c, v, rand() as r FROM part_src)
            |SELECT c1.a FROM cte c1 JOIN cte c2 ON c1.a = c2.a AND c1.b = c2.b
            |""".stripMargin)
        assert(cteReuseRelations(df).isEmpty,
          "Expected no CTEReuseRelation when the guaranteed-reuse flag is off.\n" +
            df.queryExecution.optimizedPlan.treeString)
        checkAnswer(df, Seq(Row(1), Row(2), Row(3), Row(4), Row(5)))
      }
    }
  }
}

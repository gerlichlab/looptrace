import Dependencies._

ThisBuild / scalaVersion     := "3.4.2"
ThisBuild / version          := "0.6.0"
ThisBuild / organization     := "at.ac.oeaw.imba.gerlich"
ThisBuild / organizationName := "Gerlich Group, IMBA, OEAW"

/* sbt-github-actions settings */
ThisBuild / githubWorkflowOSes := Seq("ubuntu-latest", "ubuntu-20.04")
ThisBuild / githubWorkflowTargetBranches := Seq("main")
ThisBuild / githubWorkflowPublishTargetBranches := Seq()
ThisBuild / githubWorkflowJavaVersions := Seq("11", "17", "19", "21").map(JavaSpec.temurin)

ThisBuild / assemblyMergeStrategy := {
  // This works for the moment, but seems dangerous; what if we really needed what was in here?
  // Conflict comes from logback-classic + logback-core, each having module-info.class.
  case "module-info.class" => MergeStrategy.discard
  case x =>
    val oldStrategy = (ThisBuild / assemblyMergeStrategy).value
    oldStrategy(x)
}

lazy val root = (project in file("."))
  .settings(
    name := "looptrace",
    scalacOptions ++= Seq(
      "-deprecation",
      "-encoding", "utf8",
      //"-explain",
      "-feature",
      "-language:existentials",
      // https://contributors.scala-lang.org/t/for-comprehension-requires-withfilter-to-destructure-tuples/5953
      "-source:future", // for tuples in for comprehension; see above link
      "-unchecked",
      "-Werror",
    ),
    libraryDependencies ++= Seq(
      catsCore,
      mouse,
      os, 
      scalaCsv,
      scopt,
      uPickle,
      ) ++ 
      logging ++
      Seq( // only for tests
        scalacheck, 
        scalactic, 
        scalatest, 
        scalatestScalacheck
      ).map(_ % Test), 
  )

import Dependencies._

ThisBuild / scalaVersion     := "3.3.1"
ThisBuild / version          := "0.2.0-SNAPSHOT"
ThisBuild / organization     := "at.ac.oeaw.imba.gerlich"
ThisBuild / organizationName := "Gerlich Group, IMBA, OEAW"

/* sbt-github-actions settings */
ThisBuild / githubWorkflowOSes := Seq("ubuntu-22.04", "ubuntu-20.04")
ThisBuild / githubWorkflowPublishTargetBranches := Seq()
val JDK11 = JavaSpec.temurin("11")
val JDK17 = JavaSpec.temurin("17")
ThisBuild / githubWorkflowJavaVersions := Seq(JDK11, JDK17)

Test / testOptions += Tests.Argument("-oF")

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
    ) ++ Seq( // only for tests
      scalacheck, 
      scalactic, 
      scalatest, 
      scalatestScalacheck
    ).map(_ % Test), 
  )


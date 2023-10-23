import Dependencies._

ThisBuild / scalaVersion     := "3.3.0"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "at.ac.oeaw.imba.gerlich"
ThisBuild / organizationName := "Gerlich Group, IMBA, OEAW"

lazy val testDependencies = Seq(
  scalacheck, 
  scalactic, 
  scalatest, 
  scalatestScalacheck
  )

lazy val root = (project in file("."))
  .settings(
    name := "looptrace",
    scalacOptions ++= Seq(
      "-deprecation",
      "-encoding", "utf8",
      //"-explain",
      "-feature",
      "-language:implicitConversions",
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
      scopt,
      uPickle,
    ) ++ 
    testDependencies.map(_ % Test), 
  )


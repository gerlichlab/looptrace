import Dependencies._

/* Core settings */
val orgName = "at.ac.oeaw.imba.gerlich"
val projectName = "looptrace"
val rootPkg = s"$orgName.$projectName"
val primaryJavaVersion = "11"
val primaryOs = "ubuntu-latest"
val isPrimaryOsAndPrimaryJavaTest = s"runner.os == '$primaryOs' && runner.java-version == '$primaryJavaVersion'"
ThisBuild / scalaVersion     := "3.6.4"
ThisBuild / version          := "0.14.1"
ThisBuild / organization     := orgName
ThisBuild / organizationName := "Gerlich Group, IMBA, OEAW"

// Needed for ZARR (jzarr) (?)
ThisBuild / resolvers += "Unidata UCAR" at "https://artifacts.unidata.ucar.edu/content/repositories/unidata-releases/"

/* sbt-github-actions settings */
ThisBuild / githubWorkflowOSes := Seq(primaryOs, "ubuntu-20.04", "macos-latest")
ThisBuild / githubWorkflowTargetBranches := Seq("main")
ThisBuild / githubWorkflowPublishTargetBranches := Seq()
ThisBuild / githubWorkflowJavaVersions := Seq(primaryJavaVersion, "17", "19", "21").map(JavaSpec.temurin)
ThisBuild / githubWorkflowBuildPreamble ++= Seq(
  // Account for the absence of sbt in newer versions of the setup-java GitHub Action.
  WorkflowStep.Run(commands = List("brew install sbt"), cond = Some("contains(runner.os, 'macos')")),
  /* Add linting and formatting checks, but only limit to a single platform + Java combo. */
  WorkflowStep.Sbt(
    List("scalafmtCheckAll"),
    name = Some("Check formatting with scalafmt"),
    cond = Some(isPrimaryOsAndPrimaryJavaTest),
  ),
  WorkflowStep.Sbt(
    List("scalafixAll --check"),
    name = Some("Lint with scalafix"),
    cond = Some(isPrimaryOsAndPrimaryJavaTest),
  ),
)

ThisBuild / assemblyMergeStrategy := {
  // This works for the moment, but seems dangerous; what if we really needed what was in here?
  // Conflict comes from logback-classic + logback-core, each having module-info.class.
  case "module-info.class" => MergeStrategy.discard
  case x =>
    val oldStrategy = (ThisBuild / assemblyMergeStrategy).value
    oldStrategy(x)
}

lazy val root = (project in file("."))
  .enablePlugins(BuildInfoPlugin)
  .settings(
    name := projectName,
    buildInfoKeys := Seq[BuildInfoKey](name, version, scalaVersion, sbtVersion),
    buildInfoPackage := s"$rootPkg.internal",
    scalacOptions ++= Seq(
      "-deprecation",
      "-encoding", "utf8",
      //"-explain",
      "-feature",
      "-language:existentials",
      // https://contributors.scala-lang.org/t/for-comprehension-requires-withfilter-to-destructure-tuples/5953
      "-source:future", // for tuples in for comprehension; see above link
      "-unchecked",
      // https://www.scala-lang.org/2021/01/12/configuring-and-suppressing-warnings.html
      // Turn any compiler warning into error, then make exceptions for a couple expected messages.
      //"-Wconf:any:e,msg=old given syntax is no longer supported:w,msg=Given member definitions starting with:w",
      "-rewrite", 
      "-new-syntax",
      "-Werror",
    ),
    libraryDependencies ++= Seq(
      catsCore,
      iron, 
      ironCats, 
      mouse,
      os, 
      pureconfigCore,
      pureconfigGeneric,
      scopt,
      squants,
      uPickle,
      ) ++ 
      gerlibs ++
      logging ++
      Seq( // only for tests
        catsLaws,
        gerlibTesting,
        ironScalacheck,
        scalaCsv,
        scalacheck, 
        scalacheckOps,
        scalactic, 
        scalatest, 
        scalatestScalacheck
      ).map(_ % Test), 
  )

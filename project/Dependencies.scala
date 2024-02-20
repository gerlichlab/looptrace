import sbt._

/** Dependencies for the project */
object Dependencies {
    /* Versions */
    lazy val scalatestVersion = "3.2.18"
    lazy val (scalacheckMajor, scalacheckMinor) = ("1", "17")
    
    /* Core libraries */
    lazy val catsCore = "org.typelevel" %% "cats-core" % "2.10.0"
    lazy val logging = Seq(
        "ch.qos.logback" % "logback-classic" % "1.4.14", 
        "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
    )
    lazy val mouse = "org.typelevel" %% "mouse" % "1.2.2"
    // lazy val netcdf = {
    //     val version = "5.5.3"
    //     val orgName = "edu.ucar"
    //     Seq(
    //         orgName % "cdm-core" % version,
    //         orgName % "cdm-zarr" % version,
    //         orgName % "toolsUI" % "5.5.2"
    //     )
    // }
    lazy val os = "com.lihaoyi" %% "os-lib" % "0.9.2"
    lazy val scalaCsv = "com.github.tototoshi" %% "scala-csv" % "1.3.10"
    lazy val scopt = "com.github.scopt" %% "scopt" % "4.1.0"
    lazy val uJson = "com.lihaoyi" %% "ujson" % "3.1.4"
    lazy val uPickle = "com.lihaoyi" %% "upickle" % "3.1.4"

    /* Test dependencies */
    lazy val scalacheck = "org.scalacheck" %% "scalacheck" % s"$scalacheckMajor.$scalacheckMinor.0"
    lazy val scalactic = "org.scalactic" %% "scalactic" % scalatestVersion
    lazy val scalatest = "org.scalatest" %% "scalatest" % scalatestVersion
    lazy val scalatestScalacheck = "org.scalatestplus" %% s"scalacheck-$scalacheckMajor-$scalacheckMinor" % "3.2.17.0"

}

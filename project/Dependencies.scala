import sbt._

/** Dependencies for the project */
object Dependencies {
    /* Versions */
    lazy val scalatestVersion = "3.2.18"
    
    /* Core libraries */
    lazy val catsCore = "org.typelevel" %% "cats-core" % "2.10.0"
    lazy val logging = Seq(
        "ch.qos.logback" % "logback-classic" % "1.5.6", 
        "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
    )
    lazy val mouse = "org.typelevel" %% "mouse" % "1.2.3"
    lazy val os = "com.lihaoyi" %% "os-lib" % "0.9.3"
    lazy val scalaCsv = "com.github.tototoshi" %% "scala-csv" % "1.3.10"
    lazy val scopt = "com.github.scopt" %% "scopt" % "4.1.0"
    lazy val uJson = "com.lihaoyi" %% "ujson" % "3.1.4"
    lazy val uPickle = "com.lihaoyi" %% "upickle" % "3.1.4"

    /* Test dependencies */
    lazy val scalacheck = "org.scalacheck" %% "scalacheck" % "1.18.0"
    lazy val scalactic = "org.scalactic" %% "scalactic" % scalatestVersion
    lazy val scalatest = "org.scalatest" %% "scalatest" % scalatestVersion
    lazy val scalatestScalacheck = "org.scalatestplus" %% "scalacheck-1-17" % "3.2.18.0"

}

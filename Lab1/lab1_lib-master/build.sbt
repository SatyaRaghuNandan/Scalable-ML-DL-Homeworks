name := "lab1_lib"

organization := "se.kth.spark"

version := "1.0-SNAPSHOT"

scalaVersion := "2.11.1"

resolvers += "Kompics Releases" at "http://kompics.sics.se/maven/repository/"
resolvers += "Kompics Snapshots" at "http://kompics.sics.se/maven/snapshotrepository/"
//resolvers += Resolver.mavenLocal

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.0" % "provided"
libraryDependencies += "org.log4s" %% "log4s" % "1.3.3" % "provided"

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

publishMavenStyle := true
//credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")
publishTo := {
  val kompics = "kompics.i.sics.se"
  val keyFile = Path.userHome / ".ssh" / "id_rsa"
  if (version.value.trim.endsWith("SNAPSHOT"))
    Some(Resolver.sftp("SICS Snapshot Repository", kompics, "/home/maven/snapshotrepository") as("root", keyFile))
  else
    Some(Resolver.sftp("SICS Release Repository", kompics, "/home/maven/repository") as("root", keyFile))
}

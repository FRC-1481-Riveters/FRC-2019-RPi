
/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.vision.VisionThread;
import edu.wpi.cscore.CvSource;

import org.opencv.core.Mat;

import visiontargetfinder.*;

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "FOV": <camera's horizontal field of view in degrees> // optional (150 degrees if not specified)
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
   }
 */

public final class Main {
  private static String configFile = "/boot/frc.json";
  static float m_lastTargetHeading = Float.NaN;

  @SuppressWarnings("MemberName")
  public static class CameraConfig {
    public String name;
    public String path;
    public JsonObject config;
    public JsonElement streamConfig;
  }

  public static int team;
  public static boolean server;
  public static List<CameraConfig> cameraConfigs = new ArrayList<>();
  static long fieldOfView = 150;

  private Main() {
  }

  /**
   * Report parse error.
   */
  public static void parseError(String str) {
    System.err.println("config error in '" + configFile + "': " + str);
  }

  /**
   * Read single camera configuration.
   */
  public static boolean readCameraConfig(JsonObject config) {
    CameraConfig cam = new CameraConfig();

    // name
    JsonElement nameElement = config.get("name");
    if (nameElement == null) {
      parseError("could not read camera name");
      return false;
    }
    cam.name = nameElement.getAsString();

    // path
    JsonElement pathElement = config.get("path");
    if (pathElement == null) {
      parseError("camera '" + cam.name + "': could not read path");
      return false;
    }
    cam.path = pathElement.getAsString();

    // stream properties
    cam.streamConfig = config.get("stream");

    cam.config = config;

    cameraConfigs.add(cam);
    return true;
  }

  /**
   * Read configuration file.
   */
  @SuppressWarnings("PMD.CyclomaticComplexity")
  public static boolean readConfig() {
    // parse file
    JsonElement top;
    try {
      top = new JsonParser().parse(Files.newBufferedReader(Paths.get(configFile)));
    } catch (IOException ex) {
      System.err.println("could not open '" + configFile + "': " + ex);
      return false;
    }

    // top level must be an object
    if (!top.isJsonObject()) {
      parseError("must be JSON object");
      return false;
    }
    JsonObject obj = top.getAsJsonObject();

    // team number
    JsonElement teamElement = obj.get("team");
    if (teamElement == null) {
      parseError("could not read team number");
      return false;
    }
    team = teamElement.getAsInt();

    // ntmode (optional)
    if (obj.has("ntmode")) {
      String str = obj.get("ntmode").getAsString();
      if ("client".equalsIgnoreCase(str)) {
        server = false;
      } else if ("server".equalsIgnoreCase(str)) {
        server = true;
      } else {
        parseError("could not understand ntmode value '" + str + "'");
      }
    }

    // cameras
    JsonElement camerasElement = obj.get("cameras");
    if (camerasElement == null) {
      parseError("could not read cameras");
      return false;
    }
    JsonArray cameras = camerasElement.getAsJsonArray();
    for (JsonElement camera : cameras) {
      if (!readCameraConfig(camera.getAsJsonObject())) {
        return false;
      }
    }

    return true;
  }

  /**
   * Start running the camera.
   */
  public static VideoSource startCamera(CameraConfig config) {
    System.out.println("Starting camera '" + config.name + "' on " + config.path);
    CameraServer inst = CameraServer.getInstance();
    UsbCamera camera = new UsbCamera(config.name, config.path);
    MjpegServer server = inst.startAutomaticCapture(camera);

    Gson gson = new GsonBuilder().create();

    camera.setConfigJson(gson.toJson(config.config));
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);

    if (config.streamConfig != null) {
      server.setConfigJson(gson.toJson(config.streamConfig));
    }

    return camera;
  }

  public static class MyPipeline implements VisionPipeline {
    static float m_target;

    static final VisionTargetFinder targetFinder = new VisionTargetFinder();

    Object targetLock = new Object();

    long m_startingTimeStamp;

    Mat annotatedMat;

    @Override
    public void process(Mat mat) {
      float fCurrentTarget;

      m_startingTimeStamp = System.currentTimeMillis();

      fCurrentTarget = targetFinder.getVisionTargetLocation(mat);

      /*
       * Because the VisionPipeline is expected to run in a separate thread, lock
       * access to the m_target value to ensure no one else is attempting to read it
       * while this pipeline is writing it.
       */
      synchronized (targetLock) {
        m_target = fCurrentTarget;
      }

      annotatedMat = mat;
      targetFinder.annotateStream(annotatedMat);
    }

    public Mat getAnnotatedMat() {
      return annotatedMat;
    }

     public long getStartTime() {
      return m_startingTimeStamp;
    }

    public float getTarget() {
      float fCurrentTarget;

      synchronized (targetLock) {
        fCurrentTarget = m_target;
      }
      return fCurrentTarget;
    }
  }

  /**
   * Main.
   */
  public static void main(String... args) {
    if (args.length > 0) {
      configFile = args[0];
    }

    // read configuration
    if (!readConfig()) {
      return;
    }

    // start NetworkTables
    NetworkTableInstance ntinst = NetworkTableInstance.getDefault();
    if (server) {
      System.out.println("Setting up NetworkTables server");
      ntinst.startServer();
    } else {
      System.out.println("Setting up NetworkTables client for team " + team);
      ntinst.startClientTeam(team);
    }

    // start cameras
    List<VideoSource> cameras = new ArrayList<>();
    for (CameraConfig cameraConfig : cameraConfigs) {
      cameras.add(startCamera(cameraConfig));
    }

    NetworkTable visionTable = ntinst.getTable("Vision");
    NetworkTableEntry targetErrorEntry = visionTable.getEntry("targetError");
    NetworkTableEntry targetProcessingTimeEntry = visionTable.getEntry("targetProcessingTime");

    // start image processing on camera 0 if present
    if (cameras.size() >= 1) {

      CvSource outputStream = CameraServer.getInstance().putVideo("Annotated Vision", 320, 240);
      try {
        /*
         * Get the first camera's configuration JSONElement "FOV" if it exists, then
         * render it as a long. If the element isn't in the JSON /boot/frc.json file,
         * the get() will return a null, which will cause the getAsLong() to throw an
         * exception. Just use the default, initially set, value intead of what's in the
         * file.
         */
        fieldOfView = cameraConfigs.get(0).config.get("FOV").getAsLong();
      } catch (Exception e) {
        System.out.println(new String()
            .format("Couldn't understand camera's FOV configuration value (ex: FOV: 150 ). Using %d instead.\n", fieldOfView));
      }

      VisionThread visionThread = new VisionThread(cameras.get(0), new MyPipeline(), pipeline -> {
        long startTime = pipeline.getStartTime();

        float fTargetNormalizedHeading = pipeline.getTarget();
        float fRelativeTargetHeading = fTargetNormalizedHeading * (float) fieldOfView / 2.0f;
        long targetProcessingTime = System.currentTimeMillis() - startTime;

        /*
         * Tell the roborio what the target's new heading is. Also include the time it
         * took to process this picture. This way, the roboRIO can figure out where it
         * was actually facing at the time the picture was taken, and account for the
         * lag due to processing the picture
         */
        targetErrorEntry.setValue(fRelativeTargetHeading);
        targetProcessingTimeEntry.setValue(targetProcessingTime);

        System.out.println(new String().format("visionTargetError:%3.2f processingTime:%d", fRelativeTargetHeading,
            targetProcessingTime));

            outputStream.putFrame(pipeline.getAnnotatedMat());

      });

      visionThread.start();
    }

    // loop forever
    for (;;) {
      try {
        Thread.sleep(10000);
      } catch (InterruptedException ex) {
        return;
      }
    }
  }
}


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
import edu.wpi.first.networktables.EntryListenerFlags;
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
       "switched cameras": [
           {
               "name": <virtual camera name>
               "key": <network table key used for selection>
               // if NT value is a string, it's treated as a name
               // if NT value is a double, it's treated as an integer index
           }
       ]
   }
 */

public final class Main {
  private static String configFile = "/boot/frc.json";

  @SuppressWarnings("MemberName")
  public static class CameraConfig {
    public String name;
    public String path;
    public JsonObject config;
    public JsonElement streamConfig;
  }

  @SuppressWarnings("MemberName")
  public static class SwitchedCameraConfig {
    public String name;
    public String key;
  };

  public static int team;
  public static boolean server;
  public static List<CameraConfig> cameraConfigs = new ArrayList<>();
  public static List<SwitchedCameraConfig> switchedCameraConfigs = new ArrayList<>();
  public static List<VideoSource> cameras = new ArrayList<>();

  static long fieldOfView = 60;

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
   * Read single switched camera configuration.
   */
  public static boolean readSwitchedCameraConfig(JsonObject config) {
    SwitchedCameraConfig cam = new SwitchedCameraConfig();

    // name
    JsonElement nameElement = config.get("name");
    if (nameElement == null) {
      parseError("could not read switched camera name");
      return false;
    }
    cam.name = nameElement.getAsString();

    // path
    JsonElement keyElement = config.get("key");
    if (keyElement == null) {
      parseError("switched camera '" + cam.name + "': could not read key");
      return false;
    }
    cam.key = keyElement.getAsString();

    switchedCameraConfigs.add(cam);
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

    if (obj.has("switched cameras")) {
      JsonArray switchedCameras = obj.get("switched cameras").getAsJsonArray();
      for (JsonElement camera : switchedCameras) {
        if (!readSwitchedCameraConfig(camera.getAsJsonObject())) {
          return false;
        }
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

  /**
   * Start running the switched camera.
   */
  public static MjpegServer startSwitchedCamera(SwitchedCameraConfig config) {
    System.out.println("Starting switched camera '" + config.name + "' on " + config.key);
    MjpegServer server = CameraServer.getInstance().addSwitchedCamera(config.name);

    NetworkTableInstance.getDefault().getEntry(config.key).addListener(event -> {
      if (event.value.isDouble()) {
        int i = (int) event.value.getDouble();
        if (i >= 0 && i < cameras.size()) {
          server.setSource(cameras.get(i));
        }
      } else if (event.value.isString()) {
        String str = event.value.getString();
        for (int i = 0; i < cameraConfigs.size(); i++) {
          if (str.equals(cameraConfigs.get(i).name)) {
            server.setSource(cameras.get(i));
            break;
          }
        }
      }
    }, EntryListenerFlags.kImmediate | EntryListenerFlags.kNew | EntryListenerFlags.kUpdate);

    return server;
  }

  public static class MyPipeline implements VisionPipeline {
    static VisionTargetFinder.TargetInformation m_target;

    static final VisionTargetFinder targetFinder = new VisionTargetFinder();

    Object targetLock = new Object();

    long m_startingTimeStamp;

    Mat annotatedMat;

    @Override
    public void process(Mat mat) {
      VisionTargetFinder.TargetInformation fCurrentTarget;

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

    public VisionTargetFinder.TargetInformation getTarget() {
      VisionTargetFinder.TargetInformation fCurrentTarget;

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
    for (CameraConfig config : cameraConfigs) {
      cameras.add(startCamera(config));
    }

    // start switched cameras
    for (SwitchedCameraConfig config : switchedCameraConfigs) {
      startSwitchedCamera(config);
    }

    NetworkTable visionTable = ntinst.getTable("Vision");
    NetworkTableEntry targetErrorEntry = visionTable.getEntry("targetError");
    NetworkTableEntry targetProcessingTimeEntry = visionTable.getEntry("targetProcessingTime");
    NetworkTableEntry targetInformation = visionTable.getEntry("targetInformation");

    // start image processing on camera 0 if present
    if (cameras.size() >= 1) {

      CvSource outputStream = CameraServer.getInstance().putVideo("Annotated Vision", 160, 120);
      try {
        /*
         * Get the first camera's configuration JSONElement "FOV" if it exists, then
         * render it as a long. If the element isn't in the JSON /boot/frc.json file,
         * the get() will return a null, which will cause the getAsLong() to throw an
         * exception. Just use the default, initially set, value intead of what's in the
         * file.
         */

        fieldOfView = cameraConfigs.get(0).config.get("FOV").getAsLong();
        System.out.println(String.format("Set FOV to %d", fieldOfView));
      } catch (Exception e) {
        System.out.println(String.format(
            "Couldn't understand camera's FOV configuration value (ex: FOV: 150 ). Using %d instead.", fieldOfView));
      }

      VisionThread visionThread = new VisionThread(cameras.get(0), new MyPipeline(), pipeline -> {
        long startTime = pipeline.getStartTime();

        
        VisionTargetFinder.TargetInformation targetDetails = pipeline.getTarget();
        double fRelativeTargetHeading = targetDetails.normalizedCenter * (double) fieldOfView / 2.0f;
        long targetProcessingTime = System.currentTimeMillis() - startTime;
        double targetDistance = Double.NaN;


        /*
         * Check if the normalized returned heading is NaN (Not a Number). If it's Not a
         * Number, the target finder failed to find a heading and the value shouldn't be
         * used. Don't send invalid values to the RoboRIO.
         */
        if (!Double.isNaN(targetDetails.normalizedCenter)) {
          /*
           * Tell the roborio what the target's new heading is. Also include the time it
           * took to process this picture. This way, the roboRIO can figure out where it
           * was actually facing at the time the picture was taken, and account for the
           * lag due to processing the picture
           */
          targetErrorEntry.setValue(fRelativeTargetHeading);
          targetProcessingTimeEntry.setValue(targetProcessingTime);

          /*
           * To keep the information coherent (so that the heading and the time stamp are
           * coordinated) combine the numbers into a single array and send the whole
           * array to the RoboRIO together. That way, both pieces of information show up
           * at exactly the same time. An example of this output is
           * 
           * [3.14529424,150.0,1.40]
           * 
           * where the first floating point number is the heading and the second is the age of
           * the information in milliseconds.
           */

           			/*
				 * Compute the distance to target using known features of the target, the
				 * resolution and the FOV of the camera.
				 * 
				 * d = Tin*FOVpixel/(2*Tpixel*tanΘ)
				 * 
         * Where: 
         * Θ is 1/2 of the FOV
         * Tin is the actual width of the target, which is the distance between the centers of the vision targets.
         * FOVpixel is the width of the display in pixels (the horizontal resolution)
         * Tpixel is the length of the target in pixels (the distance between the centers of the vision targets in pixels)
				 *  
				 * dNormalized = FOVPixel/Tpixel
         * 
         * 
         * So, just compute the rest by multiplying dNormalized * Tin / (2*tanΘ)
         * 
         * 
				 * 
				 */
          double targetWidth = 11.267601903166458855661396068853; /* Distance between center of targets in inches */
          targetDistance = targetDetails.distanceToTargetNormalized * targetWidth
              / (2.0 * Math.tan(Math.toRadians((double) fieldOfView / 2.0)));

          targetInformation
              .setDoubleArray(new double[] { fRelativeTargetHeading, (double) targetProcessingTime, targetDistance });
        }

        System.out.println(String.format("visionTargetError:%3.1f degrees, distance %3.1f, processingTime:%d ms",
            fRelativeTargetHeading, targetDistance, targetProcessingTime));

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

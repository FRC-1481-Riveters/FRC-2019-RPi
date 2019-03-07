package visionhelper;

import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;

public class contourHelper {

	public contourHelper() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

	}

	public Vec2f getCenterLine(MatOfPoint contour) {

		return getCenterLine(contour, Imgproc.CV_DIST_L2);
	}

	public Vec2f getCenterLine(MatOfPoint contour, int distType) {
		MatOfPoint line = new MatOfPoint();

		Imgproc.fitLine(contour, line, distType, 0, 0.01, 0.01);

		return new Vec2f(line);

	}

	public RotatedRect getRotatedRectangle(MatOfPoint contour) {

		return Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
	}

	public double getArea(MatOfPoint contour) {
		return Imgproc.contourArea(contour);
	}

	public Point getCenter(MatOfPoint contour1, MatOfPoint contour2) {

		Moments m1 = Imgproc.moments(contour1);
		Moments m2 = Imgproc.moments(contour2);

		Point centerM1 = new Point(m1.get_m10() / m1.get_m00(), m1.get_m01() / m1.get_m00());
		Point centerM2 = new Point(m2.get_m10() / m2.get_m00(), m2.get_m01() / m2.get_m00());

		return getCenter(centerM1, centerM2);
	}

	public Point getCenter(Vec2f vect1, Vec2f vect2) {
		return getCenter(new Point(vect1.x, vect1.y), new Point(vect2.x, vect2.y));
	}

	public Point getCenter(Point p1, Point p2) {
		Point centerPoint = new Point();

		centerPoint.x = Math.min(p1.x, p2.x) + ((Math.max(p1.x, p2.x) - Math.min(p1.x, p2.x)) / 2.0f);
		centerPoint.y = Math.min(p1.y, p2.y) + ((Math.max(p1.y, p2.y) - Math.min(p1.y, p2.y)) / 2.0f);

		return centerPoint;
	}

	public double getLength(Point p1, Point p2) {
		return Math.sqrt(Math.pow(p2.x - p1.x, 2.0) + Math.pow(p2.y - p1.y, 2.0));
	}
}
package visiontargetfinder;

import java.util.ArrayList;
import java.util.Collections;

import org.opencv.core.*;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import visionhelper.contourHelper;
import visiontargetfilter.*;

public class VisionTargetFinder {

	VisionTargetFilter visionTargetFilter;

	contourHelper helper = new contourHelper();

	public VisionTargetFinder() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		visionTargetFilter = new VisionTargetFilter();

	}

	private class VisionTargetPair {
		public RotatedRect LTarget;
		public RotatedRect RTarget;

		public VisionTargetPair(RotatedRect LeftTarget, RotatedRect RightTarget) {
			LTarget = LeftTarget;
			RTarget = RightTarget;
		}
	}

	ArrayList<MatOfPoint> allContours;
	ArrayList<RotatedRect> goodRectangles = new ArrayList<>();
	ArrayList<VisionTargetPair> targetPairs = new ArrayList<>();
	Point m_selectedPoint;

	public void annotateStream(Mat matImage) {

		/* Draw all the contours we found in green. */
		for (int index = 0; index < allContours.size(); ++index) {
			Imgproc.drawContours(matImage, allContours, index, new Scalar(0, 255, 0));
		}

		/* Draw the good rectangles that we found in blue. */
		for (RotatedRect rRect : goodRectangles) {
			Point[] vertices = new Point[4];
			rRect.points(vertices);
			for (int j = 0; j < 4; j++) {
				Imgproc.line(matImage, vertices[j], vertices[(j + 1) % 4], new Scalar(255, 0, 0));
			}
		}

		/* Draw the line between the target pairs we found in red. */
		for (VisionTargetPair targetPair : targetPairs) {
			Imgproc.line(matImage, new Point(targetPair.LTarget.center.x, targetPair.LTarget.center.y),
					new Point(targetPair.RTarget.center.x, targetPair.RTarget.center.y), new Scalar(0, 0, 255));
		}

		if (m_selectedPoint != null) {
			/*
			 * Draw a marker in the middle of the selected target pair's connecting line in
			 * red.
			 */
			Imgproc.drawMarker(matImage, m_selectedPoint, new Scalar(0, 0, 255), Imgproc.MARKER_TILTED_CROSS);

			/* Draw a thick, vertical line through the target point in red. */
			Imgproc.line(matImage, new Point(m_selectedPoint.x, 0), new Point(m_selectedPoint.x, matImage.cols()),
					new Scalar(0, 0, 255), 5);
		}

	}

	public double getVisionTargetLocation(Mat matImage) {

		double position = Double.NaN;

		/*
		 * Process the image and look for contours that might be vision targets.
		 * 
		 */
		visionTargetFilter.process(matImage);

		goodRectangles.clear();
		targetPairs.clear();

		/*
		 * Read the list of all contours that the GRIP image processing found and store
		 * them in the "contours" list.
		 * 
		 * contours now holds ALL the contours that the GRIP pipeline could find.
		 */
		allContours = visionTargetFilter.convexHullsOutput();

		/*
		 * If the GRIP pipeline found *anything*, take a look at the contours it found
		 * and see if any of them are likely to be vision targets.
		 */
		if (!allContours.isEmpty()) {

			/*
			 * For each of the contours in the "contours" list, store the next contour into
			 * "item" and figure out if it's angled correctly to be either a left leaning or
			 * right leaning vision target.
			 */
			for (MatOfPoint item : allContours) {

				/*
				 * Determine how close to a pure rectangle the contour is. First, determine the
				 * rotated, bounding rectangle for the contour.
				 * 
				 * Then compute its area.
				 * 
				 * Then, determine the area of the contour.
				 * 
				 * Figure out which value is smaller, and divide it by the other. This ratio, if
				 * it's close to 1.0, is the best fit to a rectangle.
				 */

				double contourArea = helper.getArea(item);

				RotatedRect rectangle = helper.getRotatedRectangle(item);
				double rectangleArea = rectangle.size.area();

				double ratio = Math.min(contourArea, rectangleArea) / Math.max(contourArea, rectangleArea);

				if (ratio < 0.85) {
					// System.out.println(String.format("Rejected contour with ratio %f,
					// contour area %f, rectangle
					// %s",(float)ratio,contourArea,rectangle.toString()));
					continue;
				}

				/*
				 * Figure out the angle of the rectangle and check to see if it's canted to the
				 * right or the left at the correct angle. If it's the right angle (for a left
				 * or right vision target), save this rectangle to the "goodRectangles" list for
				 * later processing.
				 * 
				 */
				double lineAngle = getAdjustedAngle(rectangle);

				if (!isTiltedLikeVisionTarget(lineAngle)) {
					continue;
				}

				goodRectangles.add(rectangle);
			}

			/*
			 * Sort the good vectors (which are all probably vision targets) from leftmost
			 * to the rightmost so they can be paired up as a left and right vision target.
			 * 
			 * Use the X coordinate as the sorting key The lower the X coordinate, the
			 * further to the left on the screen the the vision target is.
			 */

			Collections.sort(goodRectangles, (o1, o2) -> (int) o1.center.x - (int) o2.center.x);

			/*
			 * Now pair each leftside vision target to a rightside vision target. Do this by
			 * sweeping from the one most to the left of the screen (which at the top of the
			 * sorted goodVectors list) to the right side of the screen (which is the last
			 * element of goodVectors).
			 * 
			 * If the target is tilted like the left target is tilted, remember that target
			 * in the "lastLeftTarget" reference to pair up later with the next right target
			 * we find.
			 * 
			 * Then, pair that "lastLeftTarget" with the next target that's tilted like a
			 * right target.
			 * 
			 * 1 left vision target + 1 right vision target = a targetPair.
			 * 
			 */
			RotatedRect lastLeftTarget = null;
			for (RotatedRect item : goodRectangles) {
				/*
				 * This is a left-side vision target. Remember this target and keep it in mind
				 * until we find a right-side target.
				 */
				if (isTiltedLikeLeftVisionTarget(getAdjustedAngle(item))) {
					lastLeftTarget = item;
					continue;
				}

				/*
				 * Oooooo!!! A right-side target! See if we have a left-side target to pair with
				 * it!
				 */
				if (isTiltedLikeRightVisionTarget(getAdjustedAngle(item))) {
					if (lastLeftTarget != null) {
						if (areHorizontallyAligned(lastLeftTarget, item)) {

							/*
							 * We DO have a left-side target to pair with it! Pair these two vision targets
							 * together to make a new VisionTargetPair!
							 */
							targetPairs.add(new VisionTargetPair(lastLeftTarget, item));

							/*
							 * Since we've already paired-off this left-side target reset it to null so we
							 * don't use it again. We want a new left-side target assigned to
							 * lastLeftTarget, and assigning null will help us know that we don't have a
							 * left-side target in mind yet.
							 */
							lastLeftTarget = null;
						}
					}
				}
			}

			/*
			 * Check if we found *any* vision target pairs. If we did, figure out one which
			 * one to aim the robot at.
			 * 
			 * Track to the pair that's closest to the center of the camera, as this is
			 * likely to be the one that the driver aimed at.
			 */
			if (targetPairs.size() > 0) {

				/*
				 * Initialize everything with the first target pair that we know about. First,
				 * compute the distance between the center of the targetPair and the center of
				 * the screen.
				 * 
				 * Store this distance between the center point between the vision targets and
				 * the center of the screen into the variable "leastDistanceFromCenter". (Since
				 * we don't care about the Y coordinate, just ignore that value from each of the
				 * targetPairs.)
				 * 
				 * Then, compare this leastDistanceFromCenter to the distance from the center of
				 * each of the other targetPairs.
				 * 
				 * If the next targetPair's center is closer to the center of the screen, update
				 * leastDistanceFromCenter to that targetPair's center-to-center distance, and
				 * remember that targetPair's center point in "closestCenterPoint"
				 * 
				 */
				Point closestCenterPoint = helper.getCenter(targetPairs.get(0).LTarget.center,
						targetPairs.get(0).RTarget.center);
				double leastDistanceFromCenter = Math.abs((matImage.cols() / 2) - closestCenterPoint.x);

				for (int index = 1; index < targetPairs.size(); ++index) {

					Point centerPoint = helper.getCenter(targetPairs.get(index).LTarget.center,
							targetPairs.get(index).RTarget.center);
					double distanceFromCenter = Math.abs((matImage.cols() / 2) - centerPoint.x);

					if (distanceFromCenter < leastDistanceFromCenter) {
						leastDistanceFromCenter = distanceFromCenter;

						closestCenterPoint = centerPoint;
					}
				}

				/*
				 * closestCenterPoint contains the x and y coordinates of the center of the
				 * targetPair that's closest to the center of the screen. Use this targetPair's
				 * x coordinate as the target. It's the one that the driver was probably
				 * interested it (as it's the target that's closest to the center of the screen
				 * when they enabled the auto assist.)
				 * 
				 * Normalize the distance between the targetPair's center point and the center
				 * point of the screen to a value between -1.0 and 1.0, where 0.0 means you're
				 * dead-center, -1.0 means you're all the way to the right, and 1.0 means you're
				 * all the way to the left.
				 */

				position = 2.0f * ((closestCenterPoint.x / matImage.cols()) - 0.5f);

				m_selectedPoint = closestCenterPoint;
			}
		}

		return position;

	}

	boolean isTiltedLikeVisionTarget(double angle) {

		// left vision target --------------------- Right vision target
		return (isTiltedLikeLeftVisionTarget(angle) || isTiltedLikeRightVisionTarget(angle));

	}

	boolean isTiltedLikeLeftVisionTarget(double angle) {

		return (angle < 85.7 && angle > 55.7);

	}

	boolean isTiltedLikeRightVisionTarget(double angle) {

		return (angle > 94.3 && angle < 124.3);

	}

	/*
	 * Check if the origins of these two vectors are close to horizontal with each
	 * other. They must draw a line that's mostly horizontal.
	 */
	boolean areHorizontallyAligned(RotatedRect first, RotatedRect second) {
		try {
			double lineAngle = Math
					.toDegrees(Math.atan((first.center.y - second.center.y) / (first.center.x - second.center.x)));

			if (lineAngle < -15.0 || lineAngle > 15.0) {
				/*
				 * This is not a very horizontal line. Return false indicating that these two
				 * rectangles's origins are not on a horizontal enough line.
				 */
				return false;
			}
		} catch (ArithmeticException e) {
			/*
			 * Something went wrong with the calculation. Don't use this filter to filter
			 * out a pair of rectangles' origins. Just return a fake answer.
			 */
			System.out.println(String.format("Couldn't compute horizontal line angle between %s and %s",
					first.toString(), second.toString()));
		}

		return true;
	}

	/*
	 * Rotated rectangle angles are weird. See
	 * https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-
	 * rotatedrect-minarearect/ for more details
	 * 
	 * This normalizes it to a nominal polar framework, where the 0 degrees is
	 * parallel to the x axis poing to the right. 180 is parallel to the x axis
	 * pointing to the left.
	 */
	double getAdjustedAngle(RotatedRect rRect) {
		double angle;
		if (rRect.size.width < rRect.size.height) {
			angle = 90 - rRect.angle;
		} else {
			angle = -rRect.angle;
		}

		return angle;
	}

}
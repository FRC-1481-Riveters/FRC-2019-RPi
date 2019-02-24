package visionhelper;

import org.opencv.core.MatOfPoint;

public class Vec2f {

	public double vecX;
	public double vecY;
	public double x;
	public double y;

	public Vec2f() {
	}

	public Vec2f(MatOfPoint matrix) {

		vecX = matrix.get(0, 0)[0];
		vecY = matrix.get(1, 0)[0];
		x = matrix.get(2, 0)[0];
		y = matrix.get(3, 0)[0];
	}

	public MatOfPoint get() {
		MatOfPoint matrix = new MatOfPoint();

		matrix.put(0, 0, vecX);
		matrix.put(1, 0, vecY);
		matrix.put(2, 0, x);
		matrix.put(3, 0, y);

		return matrix;
	}
	
	public double getX() {
		return x;
	}
	
	public double getY() {
		return y;
	}
	
	public double getVecX() {
		return vecX;
	}
	
	public double getVecY() {
		return vecY;
	}

	
	public float getAngle() {

		try {
			// Opposite over Adjacent
			return (float) Math.toDegrees(Math.atan(vecX / vecY));
		} catch (ArithmeticException e) {

			/*
			 * Handle situations where the atan blows up. If the vecY is 0, the vector is
			 * lying on the X axis. Determine if it's pointing left or right and assign an
			 * angle as appropriate.
			 * 
			 * Remember, the Y axis is flipped about the X axis; down is increasing Y, and
			 * up is decreasing Y.
			 */
			if (vecX > 0.0) {
				return 90;
			} else {
				return -90;
			}
		}
	}


	public String dump() {
		return (String.format("[x=%f, y=%f, vecX=%f, vecY=%f]", x, y, vecX, vecY));
	}

}

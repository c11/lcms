package net.larse.lcms.coneproj;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by yang on 2/17/15.
 *
 * This class represents the result from Cone Projection - Polar Cone.
 *
 * TODO: refactory this class with ConstraintConeProjectionResult.
 *
 * ConeProj.coneA()
 *
 */
public class PolarConeProjectionResult {
  /**
   * The dimension of the face of the constraint cone on which the projection lands.
   */
  public double df;

  /**
   * The projection of y on the constraint cone
   */
  public SimpleMatrix thetahat;

  /**
   * The number of iterations before the algorithm converges.
   */
  public int steps;

  public PolarConeProjectionResult(double df, SimpleMatrix thetahat, int steps) {
    this.df = df;
    this.thetahat = thetahat;
    this.steps = steps;
  }
}

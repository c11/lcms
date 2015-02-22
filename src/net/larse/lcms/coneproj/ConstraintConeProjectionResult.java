package net.larse.lcms.coneproj;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by yang on 2/20/15.
 *
 * This class represents the result from Cone Projection – Constraint Cone.
 *
 * TODO: refactory this class with PolarConeProjectionResult.
 *
 * ConeProj.coneB()
 *
 */
public class ConstraintConeProjectionResult {
  /**
   * The dimension of the face of the constraint cone on which the projection lands.
   */
  public double df;

  /**
   * The projection of y on the constraint cone.
   */
  public double[] yhat;

  /**
   * The coefficients of the basis of the linear space and the constraint cone edges contained in the constraint cone.
   */
  public double[] coefs;

  /**
   * The number of iterations before the algorithm converges.
   */
  public int steps;

  public PolarConeProjectionResult(double df, double[] yhat, int steps, double[] coefs) {
    this.df = df;
    this.yhat = yhat;
    this.steps = steps;
    this.coefs = coefs;
  }
}

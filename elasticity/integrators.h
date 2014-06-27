

namespace Elasiticity
{
  /**
   * The residual resulting from the linear strain-stress relationship
   * (plane strain)
   * \f{
   * \left[ \begin{array}{c}
   * \sigma_x \\ \sigma_y \\ \tau_{xy}
   * \end{array}\right]
   * = \frac{E}{1-\nu^2}
   * \left[ \begin{array}{ccc}
   * 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2}
   * \end{array}\right]
   * \left[ \begin{array}{c}
   * \epsilon_x \\ \epsilon_y \\ \gamma_{xy}
   * \end{array}\right]
   * \f}
   * with finite deformations, such that
   * \f{
   * \left[ \begin{array}{c}
   * \epsilon_x \\ \epsilon_y \\ \gamma_{xy}
   * \end{array}\right]
   * =
   * \left[ \begin{array}{c}
   * u_x +  \\ \epsilon_y \\ \gamma_{xy}
   * \end{array}\right]
   
   */
  Hooke_nonlinear_residual()
}

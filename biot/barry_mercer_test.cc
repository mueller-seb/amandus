#include <biot/barry_mercer.h>
#include <iostream>

int
main()
{

  Coefficients2Sine<2> coeff;
  BarryMercer<2> bm;

  dealii::Point<2> p0(.25, .25);
  for (double t = 0; t <= 1.; t += .01)
  {
    auto f = bm(t, p0, coeff);
    std::cout << t << '\t' << f[0] << '\t' << f[1] << '\t' << f[2] << '\t' << f[3] << '\t' << f[4]
              << std::endl;
  }

  //  const unsigned int n = 40;
  //  const double h = 1./n;
  // for (unsigned int i=0;i<=n;++i)
  //   {
  //     for (unsigned int j=0;j<=n;++j)
  // 	{
  // 	  dealii::Point<2> x(h*i, h*j);
  // 	  auto f = bm(1.7, x, coeff);
  // 	  std::cout << x(0)
  // 		    << '\t' << x(1)
  // 		    << '\t' << f[0]
  // 		    << '\t' << f[1]
  // 		    << '\t' << f[2]
  // 		    << '\t' << f[3]
  // 		    << '\t' << f[4]
  // 		    << std::endl;
  // 	}
  //     std::cout << std::endl;
  //  }
}

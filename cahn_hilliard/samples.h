#include <deal.II/base/function.h>

namespace CahnHilliard
{
  using namespace dealii;

  template <int dim>
    class Startup : public Function<dim>
  {
    public:
      Startup();
      virtual void value_list (const std::vector<Point<dim> > &points,
                               std::vector<double>   &values,
                               const unsigned int component = 0) const;
      virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                      std::vector<Vector<double> >   &values) const;
  };


  template <int dim>
    Startup<dim>::Startup ()
    :
      Function<dim>(2)
  {}


  template <int dim>
    void
    Startup<dim>::vector_value_list (
        const std::vector<Point<dim> > &points,
        std::vector<Vector<double> >   &values) const
    {
      AssertDimension(points.size(), values.size());

      const double eps_square = -1e-2;

      for(unsigned int k = 0; k < points.size(); ++k)
      {
        const Point<dim>& p = points[k];
        values[k](1) = std::sin(p(0)) * std::sin(p(1));
        values[k](0) = (values[k](1)*values[k](1) - 1)*values[k](1) + eps_square * 2.0 * values[k](1);
      }

    }


  template <int dim>
    void
    Startup<dim>::value_list (
        const std::vector<Point<dim> > &points,
        std::vector<double>   &values,
        const unsigned int component) const
    {
      AssertDimension(points.size(), values.size());

      const double eps_square = -1e-2;

      for (unsigned int k=0;k<points.size();++k)
      {
        const Point<dim>& p = points[k];
        double v = std::sin(p(0)) * std::sin(p(1));
        double l = (v*v - 1)*v + eps_square * 2.0 * v;
        if(component == 0)
        {
          values[k] = l;
        } else {
          values[k] = v;
        }
      }
    }


  class BallFunction : public Function<2>
  {
    public:
      BallFunction() : Function<2>(2) {}

      virtual double value(const Point<2>& p,
                           const unsigned int component = 0) const
      {
        if(component == 1)
        {
          return p.norm() < 0.5 ? 1.0 : -1.0;
        } else
        {
          return 0.0;
        }
      }
  };

  template <int dim>
    class CrossFunction : public dealii::Function<dim>
  {
    public:
      CrossFunction();
      virtual void value_list (const std::vector<Point<dim> > &points,
                               std::vector<double>   &values,
                               const unsigned int component = 0) const;
      virtual void vector_value_list (const std::vector<Point<dim> > &points,
                                      std::vector<Vector<double> >   &values) const;
  };


  template <int dim>
    CrossFunction<dim>::CrossFunction ()
    :
      Function<dim> (2)
  {}


  template <int dim>
    void
    CrossFunction<dim>::vector_value_list (
        const std::vector<Point<dim> > &points,
        std::vector<Vector<double> >   &values) const
    {
      AssertDimension(points.size(), values.size());

      for (unsigned int k=0;k<points.size();++k)
      {
        const Point<dim>& p = points[k];
        if (std::fabs(p(0)) < .8 && std::fabs(p(1)) < .2)
          values[k](1) = 1.;
        else if (std::fabs(p(1)) < .8 && std::fabs(p(0)) < .2)
          values[k](1) = 1.;
        else
          values[k](1) = -1.;
      }
    }


  template <int dim>
    void
    CrossFunction<dim>::value_list (
        const std::vector<Point<dim> > &points,
        std::vector<double>   &values,
        const unsigned int component) const
    {
      AssertDimension(points.size(), values.size());

      for (unsigned int k=0;k<points.size();++k)
      {
        if(component == 1)
        {
          const Point<dim>& p = points[k];
          if (std::fabs(p(0)) < .8 && std::fabs(p(1)) < .2)
            values[k] = 1.;
          else if (std::fabs(p(1)) < .8 && std::fabs(p(0)) < .2)
            values[k] = 1.;
          else
            values[k] = -1.;
        } else
        {
          values[k] = 0.0;
        }
      }
    }


  Function<2>* selector(int i)
  {
    switch(i)
    {
      case 0:
        return new Startup<2>;
      case 1:
        return new BallFunction;
      case 2:
        return new CrossFunction<2>;
    }
    return 0;
  }
}

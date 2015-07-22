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
          return p.norm() < 0.2 ? 1.0 : -1.0;
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

  class TopologicalFunction : public Function<2>
  {
    public:
      TopologicalFunction() : Function<2>(2) {}

      virtual double value(const Point<2>& p,
                           const unsigned int component = 0) const
      {
        if(component == 1)
        {
          if(p.norm() < (1.0/8.0) || p(1) < p(0)*p(0) + (1.0/8.0) - 1.0)
          {
            return 1.0;
          }
        }
        return 0.0;
      }
  };

  class TopologicalFunction2 : public Function<2>
  {
    public:
      TopologicalFunction2() : Function<2>(2) {}

      virtual double value(const Point<2>& p,
                           const unsigned int component = 0) const
      {
        if(component == 1)
        {
          if(p.norm() < (1.0/8.0) || p(1) < p(0)*p(0) + (1.0/8.0) - 1.0)
          {
            return 1.0;
          }
          return -1.0;
        }
        return 0.0;
      }
  };

  class SquareFunction : public Function<2>
  {
    public:
      SquareFunction() : Function<2>(2) {}

      virtual double value(const Point<2>& p,
                           const unsigned int component = 0) const
      {
        if(component == 1)
        {
          if(std::abs(p(0)) <= 0.15 && std::abs(p(1)) <= 0.15)
          {
            return 1.0;
          } else {
            return -1.0;
          }
        }
        return 0.0;
      }
  };

  class Breakup : public Function<2>
  {
    public:
      Breakup() : Function<2>(2) {}

      virtual double value(const Point<2>& p,
                           const unsigned int component = 0) const
      {
        if(component == 1)
        {
          if(std::abs(p(1)) <= 0.3 && std::abs(p(0)) <= p(1)*p(1) + 0.025)
          {
            return 1.0;
          } else {
            return -1.0;
          }
        }
        return 0.0;
      }
  };

  class DoubleBall : public Function<2>
  {
    public:
      DoubleBall() : Function<2>(2) {}

      virtual double value(const Point<2>& p,
                           const unsigned int component = 0) const
      {
        if(component == 1)
        {
          double r = 0.175;
          double overlap = 0.0025;
          Point<2> center1(0.0, r - overlap);
          Point<2> center2(0.0, overlap - r);
          if((p - center1).norm() <= r || (p - center2).norm() <= r)
          {
            return 1.0;
          } else {
            return -1.0;
          }
        }
        return 0.0;
      }
  };
  
  class Strip : public Function<2>
  {
    public:
      Strip() : Function<2>(2) {}

      virtual double value(const Point<2>& p,
                           const unsigned int component = 0) const
      {
        if(component == 1)
        {
          if(std::abs(p(1)) < 0.5 &&
             (std::abs(p(0)) < 0.05 ||
             ((p(0) > 0 && p(0) < -2.0*p(1)) || (p(0) < 0 && p(0) > -2.0*p(1)))
             ))
          {
            return 1.0;
          } else {
            return -1.0;
          }
        }
        return 0.0;
      }
  };

  template <int dim>
  Function<dim>* selector(int i)
  {
    switch(i)
    {
      case 0:
        return new Startup<dim>;
      case 1:
        return new BallFunction;
      case 2:
        return new CrossFunction<dim>;
      case 3:
        return new TopologicalFunction;
      case 4:
        return new TopologicalFunction2;
      case 5:
        return new ZeroFunction<dim>(2);
      case 6:
        return new SquareFunction;
      case 7:
        return new Breakup;
      case 8:
        return new DoubleBall;
      case 9:
        return new Strip;
    }
    return 0;
  }


  template <int dim>
    class ShearAdvection : public Function<dim>
  {
    public:
      ShearAdvection(double strength) : Function<dim>(dim), strength(strength) {}

      virtual double value(const Point<dim>& p,
                           const unsigned int component = 0) const
      {
        if(component == 0)
        {
          return -1.0 * this->strength * p(1);
        } else
        {
          return 0.0;
        }
      }

    private:
      const double strength;
  };

  template <int dim>
  Function<dim>* advectionselector(int i, double strength = 0.0)
  {
    switch(i)
    {
      case 0:
        return new ShearAdvection<dim>(strength);
    }
    return 0;
  }
}

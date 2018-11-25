using namespace dealii;

template <int dim>
class Solution : public Function<dim>
{
public:
  Solution();
  virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const override;
  virtual Tensor<1, dim> gradient(const Point<dim>& p, const unsigned int component = 0) const override;
  virtual double laplacian(const Point<dim>& p, const unsigned int component = 0) const override;
};

template <int dim>
Solution<dim>::Solution()
{
}

template <int dim>
double
Solution<dim>::value(const Point<dim>& p, const unsigned int) const
{
  const double x = p[0];
  const double y = p[1];

  return (x*x-1)*(y*y-1);
}

template <int dim>
double
Solution<dim>::laplacian(const Point<dim>& p, const unsigned int component) const
{
  const double x = p[0];
  const double y = p[1];
  double val = 0;

if (component == 0)
  val = 2*(y*y-1)+2*(x*x-1);
else if (component == 1)
  val = -2;

  return val;
}

template <int dim>
Tensor<1, dim>
Solution<dim>::gradient(const Point<dim>& p, const unsigned int component) const
{
  Tensor<1, dim> val;
  const double x = p[0];
  const double y = p[1];

  val[0] = 2*x*(y*y-1);

if (component == 0)
  val[1] = 2*y*(x*x-1);
else if (component == 1)
  val[1] = 0;

  return val;
}

template <int dim>
void
Solution<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                      const unsigned int) const
{
  Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));

  for (unsigned int i = 0; i < points.size(); ++i)
  {
    const Point<dim>& p = points[i];
    values[i] = value(p);
  }
}

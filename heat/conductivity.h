using namespace dealii;

template <int dim>
class Conductivity : public Function<dim>
{
public:
  Conductivity(const double margin = 0.0);
  virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                          const unsigned int component = 0) const override;
  private:
    const double margin; //MARGIN between low dimensional embedding/pole and boundaries
};

template <int dim>
Conductivity<dim>::Conductivity(const double margin) : margin(margin)
{
}

template <int dim>
double Conductivity<dim>::value(const Point<dim>& p, const unsigned int component) const
{
  double x = p(0);
  double y = p(1);
  bool onEmbedding = (abs(y) < 1e-5) && (abs(x) <= (1-margin));
  double result = 1e-3; //not on face

  if (component == 1) //on face
    {
    if (onEmbedding)
      result = 1e-3; //on embedding
    else
      result = 0; //not on embedding
    }
  return result;
}

template <int dim>
void Conductivity<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
                         const unsigned int component) const
{
  AssertDimension(points.size(), values.size());

  for (unsigned int k = 0; k < points.size(); ++k)
  {
    const Point<dim>& p = points[k];
    values[k] = value(p, component);
  }
}

#ifndef __testing_heat_h
#define __testing_heat_h

using namespace dealii;

template <int dim>
class QPointsOut
{
private:
	const std::vector<Point<dim>>& qp1;
        const std::vector<Point<dim>>& qp2;
	unsigned int n1 = 0;
	unsigned int n2 = 0;
	const double eps = 1e-5;
	bool y0 = true;
public:
	QPointsOut(const FEValuesBase<dim>& fe1, const FEValuesBase<dim>& fe2);
	void write();
};

template <int dim>
QPointsOut<dim>::QPointsOut(const FEValuesBase<dim>& fe1, const FEValuesBase<dim>& fe2):
qp1(fe1.get_quadrature_points()), qp2(fe2.get_quadrature_points())
{
	n1 = qp1.size();
	n2 = qp2.size();
	Assert((n1 == n2), ExcInternalError());
}

template <int dim>
void QPointsOut<dim>::write()
{
std::vector<double> x1(n1), x2(n2), y1(n1), y2(n2);
for (int i = 0; i < n1; i++)
	{
	//Point<dim> pt = qp1[i];
	y1[i] = qp1[i](1);
	y2[i] = qp2[i](1);
	x1[i] = qp1[i](0);
	x2[i] = qp2[i](0);
	if ((abs(y1[i])>eps) or (abs(y2[i])>eps))
		{ y0 = false; }
	}
if (y0)
	{
	std::ostringstream message;
	message << "new face with quadrature points" << std::endl;

	for (int i = 0; i < n1; i++)
		message << "(" << x1[i] << ", " << y1[i] << ") - (" << x2[i] << ", " << y2[i] << ")" << std::endl;
	deallog << message.str();
	}
}


#endif

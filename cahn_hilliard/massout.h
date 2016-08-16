#ifndef __cahn_hilliard_massout_h
#define __cahn_hilliard_massout_h

#include <deal.II/algorithms/any_data.h>
#include <deal.II/algorithms/operator.h>
#include <deal.II/base/config.h>
#include <deal.II/base/event.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <amandus/adaptivity.h>

namespace CahnHilliard
{
using namespace dealii;
using namespace Algorithms;

template <class VECTOR, int dim, int spacedim = dim>
class MassOutputOperator : public OutputOperator<VECTOR>
{
public:
  MassOutputOperator(const std::string filename_base = std::string("output"),
                     const unsigned int digits = 3);

  void parse_parameters(ParameterHandler& param);
  void initialize(const DoFHandler<dim, spacedim>& dof_handler);
  void
  initialize(Remesher<VECTOR, dim>* remesher)
  {
    this->remesher = remesher;
  }

  void assemble_mean_operator();
  void log_mass(const VECTOR* solution);
  void write_output(const AnyData& vectors);

  virtual OutputOperator<VECTOR>& operator<<(const AnyData& vectors);

protected:
  Vector<double> mean_values;
  std::vector<bool> cdofmask;
  SmartPointer<const DoFHandler<dim, spacedim>, MassOutputOperator<VECTOR, dim, spacedim>> dof;
  Remesher<VECTOR, dim>* remesher;

  const std::string filename_base;
  const unsigned int digits;

  DataOut<dim> out;

  Timer timer;
};

template <class VECTOR, int dim, int spacedim>
inline void
MassOutputOperator<VECTOR, dim, spacedim>::initialize(const DoFHandler<dim, spacedim>& dof_handler)
{
  dof = &dof_handler;
}

template <class VECTOR, int dim, int spacedim>
MassOutputOperator<VECTOR, dim, spacedim>::MassOutputOperator(const std::string filename_base,
                                                              const unsigned int digits)
  : filename_base(filename_base)
  , digits(digits)
{
  out.set_default_format(DataOutBase::gnuplot);
}

template <class VECTOR, int dim, int spacedim>
void
MassOutputOperator<VECTOR, dim, spacedim>::parse_parameters(ParameterHandler& param)
{
  out.parse_parameters(param);
}

template <class VECTOR, int dim, int spacedim>
void
MassOutputOperator<VECTOR, dim, spacedim>::assemble_mean_operator()
{
  ConstraintMatrix hanging_node_constraints;
  DoFTools::make_hanging_node_constraints(*(this->dof), hanging_node_constraints);
  hanging_node_constraints.close();

  MeshWorker::IntegrationInfoBox<dim> info_box;
  UpdateFlags update_flags = update_values;
  info_box.add_update_flags_cell(update_flags);
  info_box.initialize(
    this->dof->get_fe(), StaticMappingQ1<dim, spacedim>::mapping, &(this->dof->block_info()));

  MeshWorker::DoFInfo<dim> dofinfo(this->dof->block_info());

  AnyData out;
  mean_values.reinit(this->dof->n_dofs());
  out.add(&mean_values, "mean");

  MeshWorker::Assembler::ResidualSimple<Vector<double>> assembler;
  assembler.initialize(hanging_node_constraints);
  assembler.initialize(out);

  Integrators::MeanIntegrator<dim> integrator;

  MeshWorker::integration_loop<dim, dim>(
    this->dof->begin_active(), this->dof->end(), dofinfo, info_box, integrator, assembler);

  const FiniteElement<dim>& fe_sys = this->dof->get_fe();
  ComponentMask cmask(fe_sys.n_components(), false);
  cmask.set(1, true);
  cdofmask.resize(this->dof->n_dofs());
  DoFTools::extract_dofs(*(this->dof), cmask, cdofmask);
}

template <class VECTOR, int dim, int spacedim>
void
MassOutputOperator<VECTOR, dim, spacedim>::log_mass(const VECTOR* solution)
{
  double mass = 0.0;
  for (unsigned int i = 0; i < this->cdofmask.size(); ++i)
  {
    if (this->cdofmask[i])
    {
      mass += (*solution)[i] * this->mean_values[i];
    }
  }
  deallog << "Mass(" << this->step << "): " << mass << std::endl;
}

template <class VECTOR, int dim, int spacedim>
void
MassOutputOperator<VECTOR, dim, spacedim>::write_output(const AnyData& data)
{
  Assert((dof != 0), ExcNotInitialized());
  out.attach_dof_handler(*dof);
  for (unsigned int i = 0; i < data.size(); ++i)
  {
    const VECTOR* p = data.try_read_ptr<VECTOR>(i);
    if (p != 0)
    {
      out.add_data_vector(*p, data.name(i));
    }
  }
  std::ostringstream streamOut;
  streamOut << filename_base << std::setw(digits) << std::setfill('0') << this->step
            << out.default_suffix();
  std::ofstream out_filename(streamOut.str().c_str());
  out.build_patches();
  out.write(out_filename);
  out.clear();
}

template <class VECTOR, int dim, int spacedim>
OutputOperator<VECTOR>&
MassOutputOperator<VECTOR, dim, spacedim>::operator<<(const AnyData& data)
{
  timer.stop();
  deallog << "Time(" << this->step << "): " << timer.wall_time() << std::endl;

  write_output(data);

  if (this->mean_values.size() != this->dof->n_dofs())
  {
    assemble_mean_operator();
  }
  log_mass(data.try_read_ptr<VECTOR>(0));
  if (this->remesher != 0)
  {
    this->remesher->operator()(const_cast<AnyData&>(data), AnyData());
  }

  timer.restart();

  return *this;
}
}

#endif

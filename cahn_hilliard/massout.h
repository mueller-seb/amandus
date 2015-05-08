#ifndef __cahn_hilliard_massout_h
#define __cahn_hilliard_massout_h

#include <deal.II/base/config.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/event.h>
#include <deal.II/base/timer.h>
#include <deal.II/algorithms/operator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

namespace CahnHilliard
{
  using namespace dealii;
  using namespace Algorithms;

  template <class VECTOR, int dim>
  class Remesher
  {
    public:
      Remesher() : report(0)
      {
      }

      ~Remesher()
      {
        if(transfer != 0)
        {
          delete transfer;
        }
      }

      void init(AmandusApplicationSparse<dim>* app)
      {
        this->app = app;
        this->dofh = &(app->dofs());
        transfer = new SolutionTransfer<dim, VECTOR>(*(this->dofh));
        connect_transfer();
      }

      void init(Operator<VECTOR>* report)
      {
        this->report = report;
      }

      void remesh(AnyData& data)
      {
        deallog.push("Remesher");
        deallog << "Remeshing..." << std::endl;
        //VECTOR* solution = data.entry("solution");
        assess(data);
        deallog << "Total criterion: " << this->criterion.block(0).l2_norm()
          << std::endl;
        extract_vectors(data);
        remesh();
        deallog.pop();
      }

      void extract_vectors(const AnyData& data)
      {
        to_transfer.resize(0);
        for(unsigned int i = 0; i < data.size(); ++i)
        {
          if(data.is_type<VECTOR*>(i))
          {
            originals.push_back(data.entry<VECTOR*>(i));
            to_transfer.push_back(*(data.try_read_ptr<VECTOR>(i)));
          }
        }
        deallog << "Extracted " << to_transfer.size() << " vectors." << std::endl;
      }

      void assess(const AnyData& data)
      {
        Integrators::H1ErrorIntegrator<dim> h1_error_integrator;
        ZeroFunction<dim> zero(2);
        ErrorIntegrator<dim> error_integrator(zero);
        ComponentMask mask(2, false);
        mask.set(1, true);
        error_integrator.add(&h1_error_integrator, mask);
        this->app->error(criterion, data, error_integrator);
      }

      void connect_transfer()
      {
        this->app->signals.pre_refinement.connect(
            std_cxx11::bind(&Remesher<VECTOR, dim>::prepare_transfer,
                            std_cxx11::ref(*this)));
        this->app->signals.post_refinement.connect(
            std_cxx11::bind(&Remesher<VECTOR, dim>::finalize_remeshing,
                            std_cxx11::ref(*this)));
      }

      void remesh()
      {
        this->app->refine_mesh(this->criterion.block(0), 0.1, 0.1); // TODO: adjust parameters
      }

      void prepare_transfer()
      {
        this->transfer->prepare_for_coarsening_and_refinement(to_transfer);
      }

      void finalize_remeshing()
      {
        this->app->setup_system();
        result.resize(to_transfer.size());
        for(unsigned int i = 0; i < result.size(); ++i)
        {
          result[i].reinit(this->app->dofs().n_dofs());
        }
        this->transfer->interpolate(to_transfer, result);
        for(unsigned int i = 0; i < result.size(); ++i)
        {
          deallog << "Writing back interpolated vector " << i << "." << std::endl;
          //this->app->setup_vector(
          *(originals[i]) = result[i];
        }
        if(report != 0)
        {
          deallog << "Reporting." << std::endl;
          report->notify(Events::remesh);
        }
        //this->transfer->clear();
        delete this->transfer;
        this->transfer = new SolutionTransfer<dim, VECTOR>(*(this->dofh));
      }

    protected:
      const DoFHandler<dim>* dofh;
      AmandusApplicationSparse<dim>* app;
      BlockVector<double> criterion;
      SolutionTransfer<dim, VECTOR>* transfer;
      std::vector<VECTOR*> originals;
      std::vector<VECTOR> to_transfer;
      std::vector<VECTOR> result;
      Operator<VECTOR>* report;
  };

  template <class VECTOR, int dim, int spacedim=dim>
  class MassOutputOperator : public OutputOperator<VECTOR>
  {
  public:
    MassOutputOperator(const std::string filename_base = std::string("output"),
                       const unsigned int digits = 3);

    void parse_parameters(ParameterHandler &param);
    void initialize(const DoFHandler<dim, spacedim>& dof_handler);
    void initialize(Remesher<VECTOR, dim>* remesher)
    {
      this->remesher = remesher;
    }

    void assemble_mean_operator();
    void log_mass(const VECTOR* solution);
    void write_output(const AnyData &vectors);

    virtual OutputOperator<VECTOR>& operator<<(const AnyData &vectors);

  protected:
    Vector<double> mean_values;
    std::vector<bool> cdofmask;
    SmartPointer<const DoFHandler<dim, spacedim>,
                 MassOutputOperator<VECTOR, dim, spacedim> > dof;
    Remesher<VECTOR, dim>* remesher;

    const std::string filename_base;
    const unsigned int digits;

    DataOut<dim> out;

    Timer timer;
  };

  template <class VECTOR, int dim, int spacedim>
  inline void
  MassOutputOperator<VECTOR, dim, spacedim>::initialize(
      const DoFHandler<dim, spacedim> &dof_handler)
  {
    dof = &dof_handler;
  }

  template <class VECTOR, int dim, int spacedim>
  MassOutputOperator<VECTOR, dim, spacedim>::MassOutputOperator (
    const std::string filename_base,
    const unsigned int digits)
    :
    filename_base(filename_base),
    digits(digits)
  {
    out.set_default_format(DataOutBase::gnuplot);
  }


  template <class VECTOR, int dim, int spacedim>
  void
  MassOutputOperator<VECTOR, dim, spacedim>::parse_parameters(
      ParameterHandler &param)
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
    info_box.initialize(this->dof->get_fe(),
                        StaticMappingQ1<dim, spacedim>::mapping,
                        &(this->dof->block_info()));

    MeshWorker::DoFInfo<dim> dofinfo(this->dof->block_info());

    AnyData out;
    mean_values.reinit(this->dof->n_dofs());
    out.add(&mean_values, "mean");

    MeshWorker::Assembler::ResidualSimple<Vector<double> > assembler;
    assembler.initialize(hanging_node_constraints);
    assembler.initialize(out);

    Integrators::MeanIntegrator<dim> integrator;

    MeshWorker::integration_loop<dim, dim>(
        this->dof->begin_active(), this->dof->end(),
        dofinfo, info_box,
        integrator, assembler);

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
    for(int i = 0; i < this->cdofmask.size(); ++i)
    {
      if(this->cdofmask[i])
      {
        mass += (*solution)[i] * this->mean_values[i];
      }
    }
    deallog << "Mass(" << this->step << "): " << mass << std::endl;
  }

  template <class VECTOR, int dim, int spacedim>
  void
  MassOutputOperator<VECTOR, dim, spacedim>::write_output(
    const AnyData &data)
  {
    Assert((dof!=0), ExcNotInitialized());
    out.attach_dof_handler (*dof);
    for (unsigned int i=0; i<data.size(); ++i)
    {
      const VECTOR *p = data.try_read_ptr<VECTOR>(i);
      if (p!=0)
      {
        out.add_data_vector(*p, data.name(i));
      }
    }
    std::ostringstream streamOut;
    streamOut << filename_base
      << std::setw(digits) << std::setfill('0') << this->step
      << out.default_suffix();
    std::ofstream out_filename(streamOut.str().c_str());
    out.build_patches();
    out.write(out_filename);
    out.clear();
  }


  template <class VECTOR, int dim, int spacedim>
  OutputOperator<VECTOR> &
  MassOutputOperator<VECTOR, dim, spacedim>::operator<<(
    const AnyData &data)
  {
    timer.stop();
    deallog << "Time(" << this->step << "): " <<
      timer.wall_time() << std::endl;

    write_output(data);

    if(this->mean_values.size() != this->dof->n_dofs())
    {
      assemble_mean_operator();
    }
    log_mass(data.try_read_ptr<VECTOR>(0));
    if(this->remesher != 0)
    {
      this->remesher->remesh(const_cast<AnyData&>(data));
    }

    timer.restart();

    return *this;
  }
}

#endif

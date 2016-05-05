#include <fstream>
#include <deal.II/base/mpi.h>
#include "optimization.h"
#include "stokes-slip.hh"



StokesSlip *stokes_problem;

void myfunc(const alglib::real_1d_array &x, double &fi, void *ptr)
{
  fi = stokes_problem->run(x);
}

void myfunc_grad(const alglib::real_1d_array &x, double &fi, alglib::real_1d_array &grad, void *ptr)
{
  const double d = 1e-5;
  int np, rank;
  std::vector<double> fd(x.length());

  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  for (int i=0; i<x.length(); i++)
  {
    double buf[x.length()];
    for (int j=0; j<x.length(); j++)  buf[j] = x[j];
    buf[i] += d;
    MPI_Send(&buf, x.length(), MPI_DOUBLE, i%(np-1)+1, 0, MPI_COMM_WORLD);
  }
  
  fi = stokes_problem->run(x);
  
  for (int i=0; i<x.length(); i++)
  {
    double fd;
    MPI_Status status;
    MPI_Recv(&fd, 1, MPI_DOUBLE, i%(np-1)+1, 0, MPI_COMM_WORLD, &status);
    grad[i] = (fd-fi)/d;
  }
}

void report(const alglib::real_1d_array &x, double fval, void *ptr)
{
  static int iter = 0;
  static std::ofstream f(stokes_problem->prm().logfile.c_str());
  
  if (iter == 0)
    f << "iter fval x\n-------------\n";
  
  printf("\n----------------------------\n");
  printf("iter  fval\n");
  printf("%5d %g\n", ++iter, fval);
  printf("----------------------------\n\n");

  f << iter << " " << fval << " ";
  for (unsigned int i=0; i<x.length(); ++i)
    f << x[i] << " ";
  f << std::endl;
}


int main (int argc, char **argv)
{
  if (argc == 2)
  {
    MPI_Init(&argc, &argv);

    int nproc, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    stokes_problem = new StokesSlip(argv[1]);
    const unsigned int np = stokes_problem->prm().np;

    if (rank == 0)
    {
      alglib::real_1d_array x;
      double epsg = 1e-6;
      double epsf = 0;
      double epsx = 0;
      alglib::ae_int_t maxits = stokes_problem->prm().maxit;
      alglib::minbleicstate state;
      alglib::minbleicreport rep;
      x.setlength(np);
      
      // setup bound constraints
      alglib::real_1d_array lb, ub;
      lb.setlength(x.length());
      ub.setlength(x.length());
      FunctionParser<1> fpl(1), fpu(1);
      fpu.initialize("x", stokes_problem->prm().f_max, {});
      fpl.initialize("x", stokes_problem->prm().f_min, {});
      for (unsigned int i=0; i<np; i++)
      {
        x[i] = 0;
        Point<1> p(double(i)/(np-1));
        lb[i] = fpl.value(p);
        ub[i] = fpu.value(p);
        printf("lb[%d] = %f ub[%d] = %f\n", i, lb[i], i, ub[i]);
      }
      
      // setup linear constraints
      alglib::real_2d_array c;
      alglib::integer_1d_array ct;
      c.setlength(4*np-2,np+1);
      ct.setlength(4*np-2);
      // 1st order difference constraints
      for (unsigned int i=0; i<np; ++i)
      {
        c(2*i,i+1)= 1;
        c(2*i,i)  =-1;
        c(2*i,np) = stokes_problem->prm().g_max/(np-1);
        ct(2*i) = -1;
        c(2*i+1,i+1)= 1;
        c(2*i+1,i)  =-1;
        c(2*i+1,np) = -stokes_problem->prm().g_max/(np-1);
        ct(2*i+1) = 1;
      }
      // 2nd difference constraints
      for (unsigned int i=1; i<np; ++i)
      {
        c(2*(np+i-1),i+1) =  1;
        c(2*(np+i-1),i)   = -2;
        c(2*(np+i-1),i-1) = 1;
        c(2*(np+i-1),np)  = stokes_problem->prm().h_max/((np-1)*(np-1));
        ct(2*(np+i-1)) = -1;
        c(2*(np+i)-1,i+1) =  1;
        c(2*(np+i)-1,i)   = -2;
        c(2*(np+i)-1,i-1) = 1;
        c(2*(np+i)-1,np)  = -stokes_problem->prm().h_max/((np-1)*(np-1));
        ct(2*(np+i)-1) = 1;
      }
      
      //alglib::minbleiccreatef(x.length(), x, 0.000001, state);
      alglib::minbleiccreate(x.length(), x, state);
      alglib::minbleicsetbc(state, lb, ub);
      alglib::minbleicsetlc(state, c, ct);
      alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);
      alglib::minbleicsetxrep(state, true);
      alglib::minbleicoptimize(state, &myfunc_grad, report);
      alglib::minbleicresults(state, x, rep);

      printf("Optimization return code: %d\n", int(rep.terminationtype));
      stokes_problem->run(x);
      
      // send zero length buffer to signalize end
      double dummy;
      for (int r=1; r<nproc; r++)
        MPI_Send(&dummy, 0, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
    }
    else
    { // if rank != 0
      double buf[np];
      MPI_Status status;
      int buf_size = np;
      alglib::real_1d_array x;
      x.setlength(np);
      std::vector<double> fd;
      int count = 0;

      while (buf_size == np)
      {
        MPI_Recv(buf, np*sizeof(double), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_DOUBLE, &buf_size);
        if (buf_size == np)
        {
          for (int i=0; i<np; i++) x[i] = buf[i];
          fd.push_back(stokes_problem->run(x));
          count++;
          
          if ((count+1)*(nproc-1)+rank-1>=np)
          {
            for (double f : fd)
              MPI_Send(&f, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            fd.clear();
            count = 0;
          }
        }
      }
      
//      for (double f : fd)
//        MPI_Send(&f, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    delete stokes_problem;
    
    MPI_Finalize();
  }
  else
  {
    std::cout << "Usage:\n\n  " << argv[0] << " parameter_file.prm\n\nSyntax of parameter_file.prm:\n\n";
  }

  return 0;
}



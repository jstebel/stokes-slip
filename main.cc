#include <fstream>
#include "optimization.h"
#include "stokes-slip.hh"




StokesSlip *stokes_problem;

void myfunc(const alglib::real_1d_array &x, double &fi, void *ptr)
{
  fi = stokes_problem->run(x);
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
	stokes_problem = new StokesSlip(argv[1]);
    const unsigned int np = stokes_problem->prm().np;
// 	stokes_problem.run ();
    
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
    
    alglib::minbleiccreatef(x.length(), x, 0.0001, state);
    alglib::minbleicsetbc(state, lb, ub);
    alglib::minbleicsetlc(state, c, ct);
    alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);
    alglib::minbleicsetxrep(state, true);
    alglib::minbleicoptimize(state, &myfunc, report);
    alglib::minbleicresults(state, x, rep);

    printf("Optimization return code: %d\n", int(rep.terminationtype));
    stokes_problem->run(x);
    
    delete stokes_problem;
  }
  else
  {
    std::cout << "Usage:\n\n  " << argv[0] << " parameter_file.prm\n\nSyntax of parameter_file.prm:\n\n";
  }

  return 0;
}

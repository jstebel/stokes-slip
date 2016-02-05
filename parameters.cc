#include "parameters.hh"
#include <sys/stat.h>
#include <iostream>

using namespace dealii;
using namespace Parameters;



AllParameters::AllParameters(const std::string &input_file)
{
	struct stat buffer;
	if (stat(input_file.c_str(), &buffer) != 0)
	{
		std::cerr << "Cannot open input file '" << input_file << "'!" << std::endl;
		exit(1);
	}
	ParameterHandler prm;
	declare_parameters(prm);
	prm.read_input(input_file);
	parse_parameters(prm);
}

void AllParameters::declare_parameters(ParameterHandler &prm)
{
// 	Materials::declare_parameters(prm);
	BoundaryConditions::declare_parameters(prm);
// 	RightHandSide::declare_parameters(prm);
	IO::declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler &prm)
{
// 	Materials::parse_parameters(prm);
	BoundaryConditions::parse_parameters(prm);
// 	RightHandSide::parse_parameters(prm);
	IO::parse_parameters(prm);
}



// void Materials::declare_parameters(ParameterHandler &prm)
// {
// 	prm.enter_subsection("Materials");
// 		prm.enter_subsection("Matrix");
// 			prm.declare_entry("Young modulus", "0.9e9", Patterns::Double(), "Young modulus of matrix");
// 			prm.declare_entry("Poisson ratio", "0.4",   Patterns::Double(), "Poisson's ratio of matrix");
// 		prm.leave_subsection();
// 
// 		prm.enter_subsection("Fiber");
// 			prm.declare_entry("Young modulus", "19.1e9",   Patterns::Double(), "Young modulus of fiber");
// 			prm.declare_entry("Poisson ratio", "0.4",      Patterns::Double(), "Poisson's ratio of fiber");
// 			prm.declare_entry("Fiber volume ratio", "0.5", Patterns::Double(), "Volume ratio of fiber in reinforcement");
// 			prm.declare_entry("Material id", "-1", Patterns::Integer(), "Id of reinforcement material in mesh");
// 			prm.declare_entry("Use 1d mesh", "false", Patterns::Bool(), "If true, fibers are read from separate mesh file as 1D elements.");
// 		prm.leave_subsection();
// 	prm.leave_subsection();
// }
// 
// void Materials::parse_parameters(ParameterHandler &prm)
// {
// 	prm.enter_subsection("Materials");
// 		prm.enter_subsection("Matrix");
// 			Young_modulus_matrix = prm.get_double("Young modulus");
// 			Poisson_ratio_matrix = prm.get_double("Poisson ratio");
// 		prm.leave_subsection();
// 
// 		prm.enter_subsection("Fiber");
// 			Young_modulus_fiber = prm.get_double("Young modulus");
// 			Poisson_ratio_fiber = prm.get_double("Poisson ratio");
// 			Fiber_volume_ratio  = prm.get_double("Fiber volume ratio");
// 			Reinforcement_material_id = prm.get_integer("Material id");
// 			use_1d_fibers = prm.get_bool("Use 1d mesh");
// 		prm.leave_subsection();
// 	prm.leave_subsection();
// }

void BoundaryConditions::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("BoundaryConditions");
		prm.declare_entry("Velocity",    "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
        prm.declare_entry("Velocity tangent",    "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
        prm.declare_entry("Velocity normal",     "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
        prm.declare_entry("Velocity slip",       "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
		prm.declare_entry("Traction",            "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
	prm.leave_subsection();
}

std::vector<std::string> inline split_string(const std::string &source, const char *delimiter = " ", bool keepEmpty = false)
{
    std::vector<std::string> results;

    size_t prev = 0;
    size_t next = 0;

    while ((next = source.find_first_of(delimiter, prev)) != std::string::npos)
    {
        if (keepEmpty || (next - prev != 0))
        {
            results.push_back(source.substr(prev, next - prev));
        }
        prev = next + 1;
    }

    if (prev < source.size())
    {
        results.push_back(source.substr(prev));
    }

    return results;
}

std::map<int,std::string> string_to_map_int_string(const std::string &str, const char delim, const char delim_key_value)
{
	std::map<int,std::string> result;
	std::vector<std::string> v = split_string(str, ",");

	for (std::string const & s : v) {
	    auto i = s.find(delim_key_value);
	    result[atoi(s.substr(0,i).c_str())] = s.substr(i + 1);
	}

	return result;
}

void BoundaryConditions::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("BoundaryConditions");
		velocity         = string_to_map_int_string(prm.get("Velocity"),         ',', ':');
        velocity_tangent = string_to_map_int_string(prm.get("Velocity tangent"), ',', ':');
        velocity_normal  = string_to_map_int_string(prm.get("Velocity normal"),  ',', ':');
        velocity_slip    = string_to_map_int_string(prm.get("Velocity slip"),    ',', ':');
		traction         = string_to_map_int_string(prm.get("Traction"),        ',', ':');
	prm.leave_subsection();
}

// void RightHandSide::declare_parameters(ParameterHandler &prm)
// {
// 	prm.enter_subsection("RightHandSide");
// 		prm.declare_entry("Forces", "", Patterns::Map(Patterns::Integer(), Patterns::Anything()));
// 	prm.leave_subsection();
// }
// 
// void RightHandSide::parse_parameters(ParameterHandler &prm)
// {
// 	prm.enter_subsection("RightHandSide");
// 		force = string_to_map_int_string(prm.get("Forces"), ',', ':');
// 	prm.leave_subsection();
// }
// 
void IO::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("InputOutput");
// 		prm.declare_entry("Mesh", "", Patterns::FileName(), "Mesh file name");
		prm.declare_entry("Output", "", Patterns::FileName(), "Output file name base");
	prm.leave_subsection();
}

void IO::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("InputOutput");
// 		mesh_file = prm.get("Mesh");
		output_file_base = prm.get("Output");
	prm.leave_subsection();
}




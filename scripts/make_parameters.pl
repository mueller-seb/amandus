#!/bin/perl

use strict;
my @type;
my @name;
my @default;
my @parname;
my @description;
my $namespace = '';
my $guard = '__parameters_h';
my %pattern =
(
    'int' => 'Integer()',
    'double' => 'Double()',
    'bool' => 'Bool()',
    'std::string' => 'Anything()'
);

my %get =
(
    'int' => 'get_integer',
    'double' => 'get_double',
    'bool' => 'get_bool',
    'std::string' => 'get'
);

#open(my $in, "<$ARGV[0]");
#open(my $out, ">$ARGV[1]");

while(<>)
{
    if (m/^parameter\s+(\w+)\s+(\w+)\s+(\"[^\"]*\")\s+(\"[^\"]*\")\s*(\"[^\"]*\")?/)
    {
	push @type, $1;
	push @name, $2;
	push @default, $3;
	push @parname, $4;
	push @description, $5;
    }
    
    elsif (m/^namespace\s+(\w+)/)
    {
	$namespace = $1;
    }
    
    elsif (m/^include_guard\s+(\w+)/)
    {
	$guard = $1;
    }
    else
    {
	die "Error: could not parse line $,\n >> $_\n";
    }
}

print <<"EOT"
#ifndef $guard
#define $guard

#include <deal.II/base/parameter_handler.h>

EOT
    ;

print "namespace $namespace\n{\n" if ($namespace ne "");

print <<'EOT'
struct Parameters : public dealii::Subscriptor
{
    static void declare_parameters(dealii::ParameterHandler& param);
    void parse_parameters(dealii::ParameterHandler& param);
EOT
    ;

for (my $i=0; $i <= $#name; ++$i)
{
    print "    $type[$i] $name[$i];\n";
}

print <<'EOT'
};

inline
void
Parameters::declare_parameters(dealii::ParameterHandler& param)
{
EOT
    ;

for (my $i=0; $i <= $#name; ++$i)
{
    print "    param.declare_entry($parname[$i], $default[$i], dealii::Patterns::$pattern{$type[$i]});\n";
}

print << 'EOT'
}

inline
void
Parameters::parse_parameters(dealii::ParameterHandler& param)
{
EOT
    ;

for (my $i=0; $i <= $#name; ++$i)
{
    print "    $name[$i] = param.$get{$type[$i]}($parname[$i]);\n";
}

print "}\n";
print "\n}\n" if ($namespace ne '');
print "#endif\n";

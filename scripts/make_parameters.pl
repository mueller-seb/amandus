
use strict;
my @type;
my @name;
my @default;
my @parname;
my @description;

my %pattern =
(
    'int' => 'Integer()',
    'double' => 'Double()',
    'std::string' => 'Anything()'
);

my %get =
(
    'int' => 'get_integer',
    'double' => 'get_double',
    'std::string' => 'get'
);

while(<>)
{
    die "Malformed line: $_\n" unless m/(\w+)\s+(\w+)\s+(\"[^\"]*\")\s+(\"[^\"]*\")\s+(\"[^\"]*\")/;
    push @type, $1;
    push @name, $2;
    push @default, $3;
    push @parname, $4;
    push @description, $5;
}

print <<'EOT'
struct Parameters : public dealii::Subscriptor
{
    static void declare_parameters(ParameterHandler& param);
    void parse_parameters(ParameterHandler& param);
EOT
    ;

for (my $i=0; $i <= $#name; ++$i)
{
    print "    $type[$i] $name[$i];\n";
}

print <<'EOT'
};

void
Parameters::declare_parameters(ParameterHandler& param)
{
EOT
    ;

for (my $i=0; $i <= $#name; ++$i)
{
    print "    param.declare_entry($parname[$i], $default[$i], Pattern::$pattern{$type[$i]});\n";
}

print << 'EOT'
};

void
Parameters::parse_parameters(ParameterHandler& param)
{
EOT
    ;

for (my $i=0; $i <= $#name; ++$i)
{
    print "    $name[$i] = param.$get{$type[$i]}($parname[$i]);\n";
}

print "};\n";
my $step;
my $level;
my $shift = -200;
my @ev;

while(<>)
{
    $step = $1 if (m/DEAL::Step\s+(\d+)/);
    $level = $2 if (m/DEAL::Triangulation\s+(\d+)\s+cells,\s+(\d+)/);

    if (m/DEAL::Eigenvalue\s+(\d+)\s+(\S+)/)
    {
	$ev[$1] = $2+$shift;
    }
    if (m/^DEAL::.* output/ || m/^DEAL::Writing solution/)
    {
	printf "%2d", $level;
	foreach (sort { $a <=> $b } @ev)
	{
	    print " $_";
	}
	print "\n";
    }
}

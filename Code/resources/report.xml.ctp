<?xml version="1.0"?>
<report>
	<configuration>
		<file>{{CONFIG}}</file>
	</configuration>
	<nodes>
		<threads>{{THREADS}}</threads>
		<machines>{{MACHINES}}</machines>
		<depths>{{DEPTHS}}</depths>
	</nodes>
	<geometry>
		<sites>{{SITES}}</sites>
		{{#PROCESSOR}}
		<domain>
			<rank>{{RANK}}</rank><sites>{{SITES}}</sites>
		</domain>
		{{/PROCESSOR}}
	</geometry>
	<results>
		<images>{{IMAGES}}</images>
		<snapshots>{{SNAPSHOTS}}</snapshots>
		<steps>
			<total>{{STEPS}}</total>
			<per_cycle>{{STEPS_PER_CYCLE}}</per_cycle>
			<per_second>{{STEPS_PER_SECOND}}</per_second>
		</steps>
	</results>
	<checks>
		{{#DENSITIES}}
		<density_problem>
			<allowed>{{ALLOWED}}</allowed>
			<actual>{{ACTUAL}}</actual>
		</density_problem>
		{{/DENSITIES}}
		{{#UNSTABLE}}
			<stability_problem/>
		{{/UNSTABLE}}
	</checks>
	<timings>
		{{#TIMER}}
		<timer>
			<name>{{NAME}}</name>
			<local>{{LOCAL}}</local>
			<min>{{MIN}}</min>
			<mean>{{MEAN}}</mean>
			<max>{{MAX}}</max>
			<normalisation>{{NORMALISATION}}</normalisation>
		</timer>
		{{/TIMER}}
	</timings>
</report>
'''
References:

[1] K. J. Sauer and T. Roessler, "Systematic Approaches to Ensure Correct Representation of 
	Measured Multi-Irradiance Module Performance in PV System Energy Production Forecasting 
	Software Programs," in IEEE Journal of Photovoltaics, vol. 3, no. 1, pp. 422-428, Jan. 2013, 
	doi: 10.1109/JPHOTOV.2012.2221080.

[2] https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.pvsystem.singlediode.html#id5
		-> S.R. Wenham, M.A. Green, M.E. Watt, “Applied Photovoltaics” ISBN 0 86758 909 4

[3] K. J. Sauer, T. Roessler and C. W. Hansen, "Modeling the Irradiance and Temperature Dependence 
	of Photovoltaic Modules in PVsyst," in IEEE Journal of Photovoltaics, vol. 5, no. 1, pp. 152-158, 
	Jan. 2015, doi: 10.1109/JPHOTOV.2014.2364133.
'''

##################################################################################################
############# User Interface #####################################################################

# These values are from the module datasheet.  All IV values are at STC.
datasheet = {'Isc':9.69, 'Voc':47, 'Imp':9.17, 'Vmp':38.7, 'Ns':72, 'u_sc':0.005} #u_sc unit is A/dC

# CSV file with irradiance and temperature test conditions and resulting Pmp value.
# Column headers in CSV file must be G, T, Pmp
data_file = 'gtp.csv'

# Number of times the fitting algorithm should run.  The Levenberg-Marquardt algorithm will
# provide a local minima.  Running the fit multiple times with randomized starting points
# can help find the global minimum but will not guarantee the global minimum is found.
runs = 3

# Parameters that are being solved for
# value = initial guess
# min = minimum of the range the parameter can be for the solution
# max = maximum of the range the parameter can be for the solution
# vary = parameter will be solved for if True.  parameter will not change form initial guess if False
# rand = if True the initial guess will be randomized uniformly within the range
u_n=		{'value':0, 'min':-0.001, 'max':0.001, 'vary':True, 'rand':True} #diode ideality temp co
Rs=			{'value':0, 'min':0.1, 'max':0.6, 'vary':True, 'rand':True} #series resistance
Rsh_ref=	{'value':0, 'min':100, 'max':1000, 'vary':True, 'rand':True} #shunt resistance at 1000 W/m^2
Rsh_0=		{'value':0, 'min':1000, 'max':10000, 'vary':True, 'rand':True} #shunt resistance at 0 W/m^2
Rsh_exp=	{'value':0, 'min':5, 'max':6, 'vary':True, 'rand':True} #shunt resistance exponential factor
# NOTES	if the min and max values for Rs and Rsh_ref are not realistic the diode ideality will be outside
#		the solution range of [0, 2] and there will be an error.
#		
#		if fitting HIT try increasing Rsh_ref.min to 200 if the above error is happening
#		if fitting CdTe try insteasing Rs.max to 2.28, Rsh_ref.min to 2000, and rest of Rsh parapemters accordingly if the above error is happening

##################################################################################################

from pvlib.pvsystem import calcparams_pvsyst, singlediode
from lmfit import Minimizer, Parameters, report_fit
from numpy import exp, random
from scipy.optimize import root_scalar
from pandas import read_csv

from time import time
then = time()
##################################################################################################

def calc_ref_vals(Isc, Voc, Imp, Vmp, Ns, Rs, Rsh):
	'''
	reference conditions assumed to be STC (1000 W/m^2, 25 C)

	Isc = STC value from datasheet
	Voc = STC value from datasheet
	Imp = STC value from datasheet
	Vmp = STC value from datasheet
	Ns = cells in series from datasheet
	Rs = series resistance (solving for)
	Rsh = shunt resistance at 1000 W/m^2 (solving for)
	'''
	q = 1.6021766e-19
	k = 1.38064852e-23
	T = 298.15


	# Solve for diode ideality factor (presented as gamma in reference) using the method described in eq7 from [1].
	# NOTE		This could be done faster if the solution range of [0.5,2] is reduced.
	def f(n):
		return ((1-exp(q*(Imp*Rs-(Voc-Vmp))/Ns/k/T/n))/(1-exp(q*(Isc*Rs-Voc)/Ns/k/T/n))-(Imp*(Rsh+Rs)-(Voc-Vmp))/(Isc*(Rsh+Rs)-Voc))
	sol = root_scalar(f, bracket=[0,2], method='brentq')
	n_ref = sol.root


	'''
	Single Diode pv cell model circuit from [2]
		I=IL-I0*(exp((V+I*Rs)/(n*Ns*Vth))-1)-(V+I*Rs)/Rsh

	rearrange to take the form y=mx+b
	IL = I0*(exp((V+I*Rs)/(n*Nsc*Vth))-1) + (V+I*Rs)/Rsh + I

	Use Isc and Voc at STC to make 2 equations with 2 unknowns
	'''
	nNsVth_ref = n_ref*Ns*k*T/q

	m1=exp((0+Isc*Rs)/(nNsVth_ref))-1
	b1=(0+Isc*Rs)/Rsh+Isc
	m2=exp((Voc+0*Rs)/(nNsVth_ref))-1
	b2=(Voc+0*Rs)/Rsh+0

	I0_ref = (b2-b1)/(m1-m2)
	IL_ref = m1*I0_ref+b1

	return I0_ref, IL_ref, n_ref

##################################################################################################

def singlediodePVSYST(G, T, I0_ref, IL_ref, n_ref, Ns, u_sc, u_n, Rs, Rsh_ref, Rsh_0, Rsh_exp):
	'''
	reference conditions assumed to be STC (1000 W/m^2, 25 C)

	G = module irradiance under test
	T = cell temperature under test
	I0_ref = diode saturation current at STC (from calc_ref_vals)
	IL_ref = cell photo current at STC (from calc_ref_vals)
	n_ref = diode ideality factor at STC (from calc_ref_vals)
	Nsc = number of cells in series (datasheet)
	u_sc = short circuit current temperature coefficient at 1000W/M^2 (datasheet)
	u_n = diode ideality factor temperature coefficient (solving for)
	Rs = series resistance (solving for)
	Rsh_ref = shunt resistance at 1000 W/m^2 (solving for)
	Rsh_0 = shunt resistance at 0 W/m^2 (solving for)
	Rsh_exp = shunt resistance exponential factor (solving for)
	'''

	IL, I0, Rs, Rsh, nNsVth = calcparams_pvsyst(G, T, u_sc, n_ref, u_n, IL_ref, I0_ref, Rsh_ref, Rsh_0, Rs, Ns, Rsh_exp,
					EgRef=1.121, irrad_ref=1000, temp_ref=25)

	out = singlediode(IL, I0, Rs, Rsh, nNsVth,
					ivcurve_pnts=None, method='lambertw')

	return(out['p_mp'])

##################################################################################################

def fcn2min(params, data, ds):
	'''
	params = parameters to minimize as a lmfit Parameters object (u_n, Rs, Rsh_ref, Rsh_0, Rsh_exp)
	data = dataframe with G, T, and Pmp flash test values
	ds  = dictionary with datasheet values for diode model

	This function returns the difference between the measured and the modeled Pmp
	'''

	v = params.valuesdict()

	I0_ref, IL_ref, n_ref = calc_ref_vals(ds['Isc'], ds['Voc'], ds['Imp'], ds['Vmp'], ds['Ns'], v['Rs'], v['Rsh_ref'])
	model = singlediodePVSYST(data['G'], data['T'], I0_ref, IL_ref, n_ref, ds['Ns'], ds['u_sc'], v['u_n'], v['Rs'], v['Rsh_ref'], v['Rsh_0'], v['Rsh_exp'])
	
	return model - data['Pmp']

##################################################################################################

data = read_csv(data_file)

for ii in range(runs):
	
	print('Fit #'+str(ii+1)+':')

	if u_n['rand'] == True : u_n['value'] = random.uniform(u_n['min'], u_n['max'])
	if Rs['rand'] == True : Rs['value'] = random.uniform(Rs['min'], Rs['max'])
	if Rsh_ref['rand'] == True : Rsh_ref['value'] = random.uniform(Rsh_ref['min'], Rsh_ref['max'])
	if Rsh_0['rand'] == True : Rsh_0['value'] = random.uniform(Rsh_0['min'], Rsh_0['max'])
	if Rsh_exp['rand'] == True : Rsh_exp['value'] = random.uniform(Rsh_exp['min'], Rsh_exp['max'])

	print('[[Initial Values]]')
	print('    u_n      = ' +str(u_n['value']))
	print('    Rs       = ' +str(Rs['value']))
	print('    Rsh_ref  = ' +str(Rsh_ref['value']))
	print('    Rsh_0    = ' +str(Rsh_0['value']))
	print('    Rsh_exp  = ' +str(Rsh_exp['value']))

	params = Parameters()
	params.add('u_n',		value=u_n['value'], min=u_n['min'], max=u_n['max'], vary=u_n['vary'])
	params.add('Rs',		value=Rs['value'], min=Rs['min'], max=Rs['max'], vary=Rs['vary'])
	params.add('Rsh_ref',	value=Rsh_ref['value'], min=Rsh_ref['min'], max=Rsh_ref['max'], vary=Rsh_ref['vary'])
	params.add('Rsh_0',		value=Rsh_0['value'], min=Rsh_0['min'], max=Rsh_0['max'], vary=Rsh_0['vary'])
	params.add('Rsh_exp',	value=Rsh_exp['value'], min=Rsh_exp['min'], max=Rsh_exp['max'], vary=Rsh_exp['vary'])

	'''
	Minimization is done as described in [3] using the G-T Matrix model.
	The 'leastsq' method is Levenberg-Marquardt.
	'''
	minner = Minimizer(fcn2min, params, fcn_args=(data, datasheet))
	result = minner.minimize(method='leastsq')

	report_fit(result)
	print('')

print(time() - then)

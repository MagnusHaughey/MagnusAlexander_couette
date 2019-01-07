


#================== Define global constants ===================#

const M = 100
const EPSILON = 1.0/24.0
const JSA = 0.985497788

const VRA = 0.0330112265 
const BBETA = VRA/(1.0+VRA)

const NUS = 25.0
const NUP = 25.0
const NUL = 80.0

const PE = 10000.0
const NA = 1.0e-4

const WI = 10.62
const DT = 0.0001

const MAXITER = 100000000




#================== Define functions ===================#

Temp = fill(0.0 , M)
dTdt = fill(0.0 , M)
Es = fill(0.0 , M)
Ep = fill(0.0 , M)
El = fill(0.0 , M)
Srr = fill(0.0 , M)
Srt = fill(0.0 , M)
Stt = fill(0.0 , M)


function elementwise(vector, array)

	if (size(array)[1] != size(array)[2]) || (size(vector)[1] != size(array)[1])
		println("Dimensions not matched correctly. Exiting...")
		exit(0)
	end

	for i in 1:size(array)[2]

		for j in 1:size(vector)[1]
		    array[i,j] *= vector[j]
		end

	end

	return array

end




function EQS( StressTemp, Srr , Srt , Stt , Temp , D1 , D2 , II )

	for i in 1:M
		Es[i] = exp(NUS*( (1.0/Temp[i])-1.0 ))
	    Ep[i] = exp(NUP*( (1.0/Temp[i])-1.0 ))
	    El[i] = exp(-NUL*( (1.0/Temp[i])-1.0 ))
	end

#=
	# Unpack StressTemp vector
	for i in 1:M
		Srr[i] = StressTemp[i]
		Srt[i] = StressTemp[M+i]
		Stt[i] = StressTemp[(2*M)+i]
		Temp[i] = StressTemp[(3*M)+i]
	end
=#

	r_vector = np.dot(D1,Temp)
	for i in 1:M
		r_vector[i] /= (Temp[i])^2
	end
	Mat = D2 + np.dot( elementwise( -NUS*r_vector , II) , D1 )
	Mat *= BBETA

	for i in 1:M
		r_vector[i] = exp(-NUS*((1.0/Temp[i]) - 1.0))
	end

	Srt = np.dot(D1 , Srt)

	for i in 1:M
		r_vector[i] *= -((1.0-BBETA)*Srt[i])
	end

	# Boundary conditions
	for i in 1:M
		Mat[1,i] = 0.0
		Mat[M,i] = 0.0
	end

	Mat[1,1] = 1.0
	Mat[M,M] = 1.0
	r_vector[1] = 0.0
	r_vector[M] = 1.0

	# Solve for velocity
	r_vector = lin.solve(Mat , r_vector)  # Velocity now contained in r_vector
	r_vector = np.dot(D1 , r_vector)


	dTdt = (np.dot(D2 , Temp))/PE
	for i in 1:M

		dTdt[i] += BBETA*(NA/PE)*Es[i]*r_vector[i]*r_vector[i] + (1.0-BBETA)*(NA/PE)*Srt[i]*r_vector[i]
		StressTemp[i] =  -(Temp[i]*El[i]*Srr[i]/WI) - (1.0-JSA)*Srt[i]*r_vector[i] + Srr[i]*dTdt[i]/Temp[i]
		StressTemp[(2*M)+i] = -(Temp[i]*El[i]*Stt[i]/WI) + (1.0+JSA)*Srt[i]*r_vector[i] + Stt[i]*dTdt[i]/Temp[i]
		StressTemp[M+i] = Temp[i]*El[i]*Srt[i]/WI + 0.5*( -(1.0-JSA)*Stt[i] + (1.0+JSA)*Srr[i] )*r_vector[i] - Srt[i]*dTdt[i]/Temp[i] + Temp[i]*r_vector[i]*Ep[i]*El[i]
		StressTemp[(3*M)+i] = BBETA*(NA/PE)*Es[i]*r_vector[i]*r_vector[i] + (1.0-BBETA)*(NA/PE)*Srt[i]*r_vector[i]
		

	end

	return StressTemp

end







#================== Import Python/NumPy modules ===================#
println("")
print("Importing Python/NumPy modules...")
using PyCall
@pyimport numpy as np
@pyimport numpy.linalg as lin

print("\rImporting Python/NumPy modules... done.")





#================== Pre-amble ===================#


II = fill(0.0 , (M,M))
ygl = fill(0.0 , M)
D1 = fill(0.0 , (M,M))
D2 = fill(0.0 , (M,M))
cbar = fill(0.0 , M)

Lstar = fill(0.0 , (M,M))
Lnplus1 = fill(0.0 , (M,M)) 

r_vector = fill(0.0 , M) 
rhsvec = fill(0.0 , M) 
T_dot = fill(0.0 , M) 
Mat = fill(0.0 , (M,M)) 

StressTemp = fill(0.0 , (4*M))
StressStar = fill(0.0 , (4*M))
F_StressTemp = fill(0.0 , (4*M))


# Fill vectors and matrices
for i in 1:M

	II[i,i] = 1.0	# Identity
	ygl[i] = cos((pi*(i-1))/(M-1))
	cbar[i] = 1.0

	StressTemp[(3*M)+i] = 1.0

end

cbar[1] = 2.0
cbar[M] = 2.0

# Fill D1
for i in 1:M
	for j in 1:M

		if (i != j)
			D1[i,j] = cbar[i] * ((-1.0)^((i-1)+(j-1)))/(cbar[j]*(ygl[i] - ygl[j]))
		
		elseif (i == j)
			D1[i,j] = (-0.5*ygl[i])/(1.0-(ygl[i]^2))
		end

	end
end

D1[1,1] = ((2.0*((M-1)^2)) + 1.0) / 6.0
D1[M,M] = -D1[1,1]

D1 *= 2.0
D2 = D1*D1

#=
for i in 1:M
	for j in 1:M
		print("$(D2[i,j]) ")
	end
	println("\n")
end
exit(0)
=#

# Operators for T
Lstar = D2*(-0.25*DT/PE)
Lnplus1 = D2*(-0.5*DT/PE)

Lstar += II
Lnplus1 += II

for i in 1:M

	Lstar[1,i] = 0.0
	Lstar[M,i] = 0.0

	Lnplus1[1,i] = 0.0
	Lnplus1[M,i] = 0.0

end

Lstar[1,1] = 1.0
Lstar[M,M] = 1.0
Lnplus1[1,1] = 1.0
Lnplus1[M,M] = 1.0



#================================================#
#================== Main code ===================#
#================================================#


# Create output files
outpath = string("./JL_DATA/FIELDS_" , MAXITER, "_", DT, "_", NUS, "_", NUP, "_", NUL, "_", PE, "_", NA, "_", WI, "/")
outvel = string(outpath , "VELOCITY/")
outstressXX = string(outpath , "STRESS_XX/")
outstressXY = string(outpath , "STRESS_XY/")
outstressYY = string(outpath , "STRESS_YY/")
outtemp = string(outpath , "TEMPERATURE/")
outtrace = string(outpath , "trace.dat")

if !(ispath(outpath)) mkdir(outpath) end
if !(ispath(outvel)) mkdir(outvel) end
if !(ispath(outstressXX)) mkdir(outstressXX) end
if !(ispath(outstressXY)) mkdir(outstressXY) end
if !(ispath(outstressYY)) mkdir(outstressYY) end
if !(ispath(outtemp)) mkdir(outtemp) end


println("")
println("")
println("k = 0")
for k in 1:MAXITER

	if (k == 1)
		global Srr = fill(0.0 , M)
		global Srt = fill(0.0 , M)
		global Stt = fill(0.0 , M)
		global StressTemp = fill(0.0 , (4*M))
		global StressStar = fill(0.0 , (4*M))
		global F_StressTemp = fill(0.0 , (4*M))

		for i in 1:M
			StressTemp[(3*M)+i] = 1.0
		end
	end

	#================== Predictor step ===================#
	# Unpack StressTemp vector
	for i in 1:M
		Srr[i] = StressTemp[i]
		Srt[i] = StressTemp[M+i]
		Stt[i] = StressTemp[(2*M)+i]
		Temp[i] = StressTemp[(3*M)+i]
	end
	
	F_StressTemp = EQS(F_StressTemp, Srr, Srt, Stt, Temp, D1, D2, II)

	for i in 1:M

		StressStar[i] = StressTemp[i] + 0.5*DT*F_StressTemp[i];
		StressStar[M+i] = StressTemp[M+i] + 0.5*DT*F_StressTemp[M+i];
		StressStar[(2*M)+i] = StressTemp[(2*M)+i] + 0.5*DT*F_StressTemp[(2*M)+i];
		Temp[i] = StressTemp[(3*M)+i];

	end

	r_vector = np.dot(D2 , Temp)
	r_vector *=  (0.25/PE)*DT
	r_vector += Temp
	r_vector[1] = 1.0
	r_vector[M] = 1.0

	for i in 1:M
		r_vector[i] += 0.5*DT*F_StressTemp[(3*M)+i]
	end

	#=== Am I imposing BCs on temperature correctly here? See C version ===#

	# Solve equation
	r_vector = lin.solve(Lstar , r_vector)

	# Put new "temp_star" values into StressStar array
	for i in 1:M
		StressStar[(3*M)+i] = r_vector[i]
	end




	#================== Corrector step ===================#
	# Unpack StressStar vector
	for i in 1:M
		Srr[i] = StressStar[i]
		Srt[i] = StressStar[M+i]
		Stt[i] = StressStar[(2*M)+i]
		Temp[i] = StressStar[(3*M)+i]
	end

	F_StressTemp = EQS(F_StressTemp, Srr, Srt, Stt, Temp, D1, D2, II)

	for i in 1:M

		StressTemp[i] += DT*F_StressTemp[i];
		StressTemp[M+i] += DT*F_StressTemp[M+i];
		StressTemp[(2*M)+i] += DT*F_StressTemp[(2*M)+i];
	
	end

	r_vector = np.dot(D2 , Temp)
	r_vector *=  (0.5/PE)*DT
	r_vector += Temp
	r_vector[1] = 1.0
	r_vector[M] = 1.0

	for i in 1:M
		r_vector[i] += DT*F_StressTemp[(3*M)+i]
	end

	# Solve equation again but with Lnplus1
	r_vector = lin.solve(Lnplus1 , r_vector)

	# Put new temperature values into StressTemp array
	for i in 1:M
		StressTemp[(3*M)+i] = r_vector[i]
	end

#=
	for i in 1:length(StressTemp)
		println("$(StressTemp[i])")
	end
	exit(0)
=#


	#================== Data acquisition ===================#
	if (k%500 == 0)
#=
		for i in 1:length(StressTemp)
			println("$(StressTemp[i])")
		end
		exit(0)
=#

		println("\rk = $k")

		for i in 1:M
			Srt[i] = StressTemp[M+i]
		end

		# Write norm of xy-stress to trace file
		if (k == 500)
			open(outtrace , "w") do file
				write(file , "$(k*DT) $(lin.norm(Srt))\n")
			end

		else 
			open(outtrace , "a") do file
				write(file , "$(k*DT) $(lin.norm(Srt))\n")
			end
		end

		# Solve for velocity profile
		r_vector = np.dot(D1,Temp)
		for i in 1:M
			r_vector[i] /= (Temp[i])^2
		end
		Mat = D2 + np.dot( elementwise( -NUS*r_vector , II) , D1 )
		Mat *= BBETA

		for i in 1:M
			r_vector[i] = exp(-NUS*((1.0/Temp[i]) - 1.0))
		end

		Srt = np.dot(D1 , Srt)

		for i in 1:M
			r_vector[i] *= -((1.0-BBETA)*Srt[i])
		end

		# Boundary conditions
		for i in 1:M
			Mat[1,i] = 0.0
			Mat[M,i] = 0.0
		end

		Mat[1,1] = 1.0
		Mat[M,M] = 1.0
		r_vector[1] = 0.0
		r_vector[M] = 1.0

		# Solve for velocity
		r_vector = lin.solve(Mat , r_vector)


		### Write data to files ###
		filename_vel = string(outvel , "$k.dat")
		open(filename_vel , "w") do file
			for i in 1:M
				write(file , "$(ygl[i]) $(r_vector[i])\n")
			end
		end

		filename_XX = string(outstressXX , "$k.dat")
		open(filename_XX , "w") do file
			for i in 1:M
				write(file , "$(ygl[i]) $(StressTemp[i])\n")
			end
		end

		filename_XY = string(outstressXY , "$k.dat")
		open(filename_XY , "w") do file
			for i in 1:M
				write(file , "$(ygl[i]) $(StressTemp[M+i])\n")
			end
		end

		filename_YY = string(outstressYY , "$k.dat")
		open(filename_YY , "w") do file
			for i in 1:M
				write(file , "$(ygl[i]) $(StressTemp[(2*M)+i])\n")
			end
		end


	end





end






















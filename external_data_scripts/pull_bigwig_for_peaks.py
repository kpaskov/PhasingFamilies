import subprocess



with open('peaks.txt', 'r') as f:
	for line in f:
		pieces = line.strip().split('\t')
		chrom = pieces[0]
		start = pieces[1]
		end = pieces[2]

		print(chrom, start, end)
		
		# # pull infant
		# bashCommand = './bigWigtoWig https://s3.amazonaws.com/DLPFC_n36/humanDLPFC/hg19/Infant_DLPFC_LIBD_meanExp.bw -chrom=%s -start=%s -end=%s expression/infant.%s.%s.%s.wig' % (chrom, start, end, chrom, start, end)
		# print(bashCommand)
		# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		# output, error = process.communicate()
		# print(output, error)

		# # pull fetal
		# bashCommand = './bigWigtoWig https://s3.amazonaws.com/DLPFC_n36/humanDLPFC/hg19/Fetal_DLPFC_LIBD_meanExp.bw -chrom=%s -start=%s -end=%s expression/fetal.%s.%s.%s.wig' % (chrom, start, end, chrom, start, end)
		# print(bashCommand)
		# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		# output, error = process.communicate()
		# print(output, error)

		# # pull child
		# bashCommand = './bigWigtoWig https://s3.amazonaws.com/DLPFC_n36/humanDLPFC/hg19/Child_DLPFC_LIBD_meanExp.bw -chrom=%s -start=%s -end=%s expression/child.%s.%s.%s.wig' % (chrom, start, end, chrom, start, end)
		# print(bashCommand)
		# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		# output, error = process.communicate()
		# print(output, error)

		# # pull teen
		# bashCommand = './bigWigtoWig https://s3.amazonaws.com/DLPFC_n36/humanDLPFC/hg19/Teen_DLPFC_LIBD_meanExp.bw -chrom=%s -start=%s -end=%s expression/teen.%s.%s.%s.wig' % (chrom, start, end, chrom, start, end)
		# print(bashCommand)
		# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		# output, error = process.communicate()
		# print(output, error)

		# # pull adult
		# bashCommand = './bigWigtoWig https://s3.amazonaws.com/DLPFC_n36/humanDLPFC/hg19/Adult_DLPFC_LIBD_meanExp.bw -chrom=%s -start=%s -end=%s expression/adult.%s.%s.%s.wig' % (chrom, start, end, chrom, start, end)
		# print(bashCommand)
		# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		# output, error = process.communicate()
		# print(output, error)

		# # pull 50+
		# bashCommand = './bigWigtoWig https://s3.amazonaws.com/DLPFC_n36/humanDLPFC/hg19/50plus_DLPFC_LIBD_meanExp.bw -chrom=%s -start=%s -end=%s expression/50plus.%s.%s.%s.wig' % (chrom, start, end, chrom, start, end)
		# print(bashCommand)
		# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		# output, error = process.communicate()
		# print(output, error)

		# pull DERs
		bashCommand = './bigBedtoBed https://s3.amazonaws.com/DLPFC_n36/humanDLPFC/hg19/DERs.bigBed -chrom=%s -start=%s -end=%s expression/ageBrainDERs.%s.%s.%s.wig' % (chrom, start, end, chrom, start, end)
		print(bashCommand)
		process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()
		print(output, error)

		


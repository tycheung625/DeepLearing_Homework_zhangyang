import csv
with open('beijing.csv') as in_file:
	with open('output.csv', 'w', newline='') as out_file:
		writer = csv.writer(out_file)
for row in csv.reader(in_file):
	if any(field.strip() for field in row):
		writer.writerow(row)

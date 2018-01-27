import sys
reload(sys)
sys.setdefaultencoding("utf-8")
file_in1 = open(sys.argv[1], 'r')
file_in2 = open(sys.argv[2], 'r')
file_out = open(sys.argv[3], 'w')
the_set = set()
for line in file_in1:
    if line == "\n":
        continue
    the_set.add(line.strip().split()[0])
for line in file_in2:
    if line == "\n":
        file_out.write("\n")
    text = line.strip()
    if text in the_set:
        file_out.write("UNT" + "\n")
    else :
        file_out.write(text + "\n")


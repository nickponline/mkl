S = """
mkl:all \dataseta{} mean-weights 0.06831687 0.75799278 0.38549148 0.48414342 0.19357713 0.00221604 0.00593759 0.00894204 0.00683584 0.01194432
mkl:all \datasetb{} mean-weights 0.09337772 0.78144366 0.37882597 0.46219665 0.15237229 0.00554916 0.006389 0.00836083 0.00876057 0.00750553
"""

for s in S.split("\n"):
	if s == "":
		continue
	name, st, blank, a, b, c, d, e, aa, bb, cc, dd, ee = s.split(" ")
	print "%s & %2.3f$\pm$%2.3f & %2.3f$\pm$%2.3f & %2.3f$\pm$%2.3f & %2.3f$\pm$%2.3f & %2.3f$\pm$%2.3f  \\\\" % (name, float(a), float(aa), float(b), float(bb), float(c), float(cc), float(d), float(dd), float(e), float(ee)) 
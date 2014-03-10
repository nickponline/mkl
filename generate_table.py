import django
from django.template import Template, Context
import csv

if __name__ == "__main__":

    # This line is required for Django configuration
    django.conf.settings.configure()

    # Open and read CSV file
    #fid = open("names.csv")
    #reader = csv.reader(fid)

    # Open and read template
    with open("table_template.tex") as f:
        t = Template(f.read())

    headers = ["metric", "$\\mu - 2\\sigma$", "$\\mu + 2\\sigma$", "$\\alpha$", "$\\mu$"]
    
    reader = [
        ["accuracy", 0.923315,0.925165,0.050000, 0.924240],
        ["error rate", 0.074702,0.075858,0.050000, 0.075280],
        ["bal", 0.075166,0.076834,0.050000, 0.076000],
        ["wracc", 0.846326,0.851114,0.050000, 0.848720],
        ["f1",0.919945,0.921723,0.050000, 0.920834],
        ["cross correlation", 0.851682,0.855550,0.050000, 0.853616],
        ["recall", 0.892034,0.895326,0.050000, 0.893680],
        ["precision", 0.953730,0.956805,0.050000, 0.955267],
        ["specificity", 0.953333,0.956107,0.050000, 0.954720],

            ]


    # Define context with the table data
    
    c = Context({"head": headers, "table": reader, "table_heading" : "Dataset 2"})

    # Render template
    output = t.render(c)

    

    # Write the output to a file
    with open("table.tex", 'w') as out_f:
        out_f.write(output)
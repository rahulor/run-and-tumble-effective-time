import clean
import os
import draw
def make_plots():
    draw.conc_profile()
    draw.initial_condition()
    draw.trajectory()
    draw.signal()
    draw.weights_for_grid()
    draw.effective_kernel_time()
    draw.weights_grouped_by_lambda_all()
    draw.T_eff_vs_Drot_all()
    draw.score_vs_Drot()
def figure_for_manuscript():
    draw.weights_grouped_by_lambda()
    draw.T_eff_vs_Drot()
    #
    if os.system("latex figure3.tex"):
        print('Error; no dvi output')
    else:
        print('Done!')
        os.system("dvips figure3.dvi -o figure3.eps")
        
def make_document():
    print('Generating pdf ...')
    if os.system("pdflatex doc.tex"):
        print('Error; no pdf output')
        print('-'*21,' Close doc.pdf if it is already open!')
    else:
        print('Done!')
    clean.these_extensions(['.aux', '.log', '.gz', '.dvi', '.ps'])
    clean.remove_if_exists('figure3-eps-converted-to.pdf')
def write_to_file(nfig):
    fig_ext = '.eps'
    path = 'data\doc_macro' + '.tex'
    line1 = '\\newcommand\\nfig{'+str(nfig)+'}'
    line2 = '\\newcommand\\figext{'+ fig_ext +'}'
    f = open(path, 'w')
    f.write(line1)
    f.write('\n')
    f.write(line2)
    f.close() 
def main():
    write_to_file(1)
    make_plots()
    figure_for_manuscript()
    make_document()
if __name__ == '__main__':
    main()
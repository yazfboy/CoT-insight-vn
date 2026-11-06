import openpyxl, json, collections, os
p=r'C:\Users\HP\Documents\Codex\2026-11-04\tr-ch-xu-t-t-nh\outputs\benchmark_review_20261106\benchmark_results_20261106_reviewed.xlsx'
w=openpyxl.load_workbook(p,read_only=True,data_only=True)
print('SHEETS',[(s.title,s.max_row,s.max_column) for s in w.worksheets])
s=w['4_Errors']; it=s.iter_rows(values_only=True); h=next(it); ix={v:i for i,v in enumerate(h)}; rows=list(it); ann=[r for r in rows if r[ix['Reasoning_State']] not in (None,'')]
print('ANNOTATED',len(ann),'types',collections.Counter(r[ix['Error_Type']] for r in ann),'faith',collections.Counter(r[ix['Faithfulness']] for r in ann))
print('SELECTED_SHEET',w['6_Selected_20'].max_row,w['6_Selected_20'].max_column)
print('CONF',list(w['7_Confidence_Compare'].iter_rows(min_row=4,max_row=6,values_only=True)))
w.close()

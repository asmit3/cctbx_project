x=[8,2,3]
if x[0]==0:
  print 'x[0] is 0'
elif x[0]==1:
  pass
elif x[0]==2:
  print 'wrong'
else:
  raise Exception('Datasource not recognized. Should be either a filename or a list/np.array/flex.double/flex.int, Sorry !')

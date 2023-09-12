
# print "hello world" to file
fn = 'test_file.txt'
#fn = '/work/tansey/test_file.txt'
f = open(fn, 'w')
f.write('Hello world, Michela here on 06/27/23!')
f.close()
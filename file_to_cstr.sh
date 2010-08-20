#!/bin/csh
@ num_args = $#argv
set output = $argv[$num_args]
/bin/rm -f $output
touch $output

@ i = 1
while ( $i < $num_args )
  set source = $argv[$i]
  set kernel_name = $argv[$i]:r:t
  echo "Converting $source to a cstring in $output"
  echo "const char * $kernel_name = \" >> $output
  sed 's/\\/\\\\/g' $source | sed 's/"/\\"/g' | awk '$1!="//" && $1!=".file"{printf(" \"%s\\n\"\n",$0)}END{printf(" ;\n")}' >> $output
  @ i = $i + 1
end


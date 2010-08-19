#!/bin/tcsh

set files = `echo *.h *.cpp *.cu`
rm -rf /tmp/cpp5678
mkdir /tmp/cpp5678
mkdir /tmp/cpp5678/geryon

foreach file ( $files )
	/bin/cp $file /tmp/cpp5678/$file:t:t
	# ------ Sed Replace
	sed 's/UCL_Vec/UCL_D_Vec/g' /tmp/cpp5678/$file:t:t > $file
end


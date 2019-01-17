"""
Borrowed from one of these SO answers: https://stackoverflow.com/questions/9031783/hide-all-warnings-in-ipython

Hides warnings better than `ifilterwarnings`.

"""

from IPython.display import HTML

HTML(
	'''
	<script>
	code_show_err=false; 
	function code_toggle_err() {
 		if (code_show_err){
 			$('div.output_stderr').hide();
 		} else {
 			$('div.output_stderr').show();
 		}
 	code_show_err = !code_show_err
	} 
	$( document ).ready(code_toggle_err);
	</script>
	To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.
	'''
)
import sqlite3
import pandas as pd
import time

import setup.duration_cal as duration_cal

start_time = time.time()

# Best model decision -
## If model variance is priority, look for highest R^2
## If predictive accuracy is priority, look for lowerst MSE (0 == Perfect model)

end_time = time.time()
final_time = end_time - start_time
final_duration, final_tag = duration_cal.duration_cal(final_time)

print("Script has reached end of line - It will terminate now!")
print(f"Script has run for {final_duration:.3f} {final_tag}!")

@echo off
echo LAUNCH_STARTED > "E:\CDAC BENGALORE\Project\project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter\launcher_log.txt"
cd /d "E:\CDAC BENGALORE\Project\project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter"
echo CWD_OK >> "E:\CDAC BENGALORE\Project\project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter\launcher_log.txt"
python -m streamlit run app.py --server.port 8085 --server.headless true >> "E:\CDAC BENGALORE\Project\project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter\streamlit_out.log" 2>> "E:\CDAC BENGALORE\Project\project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter\streamlit_err.log"
echo LAUNCH_DONE >> "E:\CDAC BENGALORE\Project\project_NSE_500_News_Filter_real_data\project_NSE_500_News_Filter\launcher_log.txt"


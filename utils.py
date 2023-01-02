import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity



def compute_similarity(job_feature_vector, user_feature_vector):
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_feature_vector, x), job_feature_vector)
    return list(cos_similarity_tfidf)


def get_applicant_id_info(applicant_id, df_job_view, df_experience, df_interest_position):
    try:
        position_view = ' và '.join(df_job_view[df_job_view['Applicant.ID'] == applicant_id]['Position'].values.tolist()).strip()
        company_view = ' và '.join(df_job_view[df_job_view['Applicant.ID'] == applicant_id]['Company'].values.tolist()).strip()
        city_view = ' và '.join(df_job_view[df_job_view['Applicant.ID'] == applicant_id]['City'].values.tolist()).strip()
    except:
        position_view = company_view = city_view = None
    
    try:
        position_experience = ' và '.join(df_experience[df_experience['Applicant.ID'] == applicant_id]['Position.Name'].values.tolist()).strip()
    except:
        position_experience = None

    try:
        position_interest = ' và '.join(df_interest_position[df_interest_position['Applicant.ID'] == applicant_id]['Position.Of.Interest'].values.tolist()).strip()
    except:
        position_interest = None

    print(f"Applicant Id: {applicant_id} đã bấm vào tin tức tuyển dụng trên website với \
            \nVị trí: {position_view} \
            \nTên công ty tương ứng là: {company_view} \
            \nThành phố tương ứng là: {city_view} \
            \nVị trí có kinh nghiệm: {position_experience} \
            \nVị trí mong muốn: {position_interest}")

    
def get_recommendation(top, df_job, final_df_jobs, scores, applicant_id=None):
    recommendation = pd.DataFrame()
    count = 0
    for i in top:
        if not applicant_id:
            recommendation.at[count, 'ApplicantID'] = applicant_id
        recommendation.at[count, 'JobID'] = int(final_df_jobs['Job.ID'][i])
        recommendation.at[count, 'title'] = final_df_jobs['Title'][i]
        recommendation.at[count, 'Position'] = df_job['Position'][i]
        recommendation.at[count, 'Company'] = df_job['Company'][i]
        recommendation.at[count, 'City'] = df_job['City'][i]
        recommendation.at[count, 'Job.Description'] = df_job['Job.Description'][i]
        recommendation.at[count, 'Employment.Type'] = df_job['Employment.Type'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation
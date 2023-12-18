setwd(here::here())

#### CONFIG
retrieve_data <- TRUE
osf_access_token <- paste(readLines("data-processing/osf-token.txt"), collapse="\n")

library(osfr)
library(tidyverse)
library(dplyr)
library(jsonlite)
library(devtools)
library(tidymedia)
library(tidyr)


if(retrieve_data){
osf_auth(osf_access_token)

osf_retrieve_node('gavy2') %>% 
  osf_ls_files() %>%
  osf_download(path='data', conflicts='skip', progress = TRUE)

  untar('data/json/json.tar.gz', exdir='data/json')
  untar('data/webm/webm.tar.gz', exdir='data/webm')
}

# Concatenate json files into one file
# Creates a list of files from json directory
json_files <- list.files(path='data/json',
                         pattern = "^\\w{8}.json$",
                         full.names = TRUE)

json.data <- json_files %>%
    map_dfr(fromJSON)

# Filter for trials with experimental data
data_trials <- c("webcam-type", "glasses", "browser-check", "public-videos", "research-videos")

experiment.data <- json.data %>%
  filter((task %in% data_trials) | (trial_type == "html-video-response"))

metadata <- experiment.data %>% 
  select(subject_id) %>% 
  unique()

# 0 = integrated webcam
# 1 = external webcam

webcam <- experiment.data %>% 
    group_by(subject_id) %>% 
    filter(task == "webcam-type") %>% 
    select("response") %>% 
    rename("webcam" = "response")


metadata <- left_join(webcam, metadata, by="subject_id")

# 0 = did not wear glasses during experiment
# 1 = wore glasses during experiment

glasses <- experiment.data %>% 
    group_by(subject_id) %>% 
    filter(task == "glasses") %>% 
    select("response") %>% 
    rename("glasses" = "response")
  
metadata <- left_join(glasses, metadata, by="subject_id")

aspect_ratio <- experiment.data %>% 
    group_by(subject_id) %>% 
    filter(task == "browser-check") %>% 
    select("aspect_ratio", "width", "height")
  
metadata <- left_join(aspect_ratio, metadata, by="subject_id")

# Webcam video resolution

metadata <- metadata %>% 
  group_by(subject_id) %>% 
  mutate(vid_width = get_width(sprintf("data/webm/%s_5_5.webm", subject_id)),
         vid_height = get_height(sprintf("data/webm/%s_5_5.webm", subject_id)))

# 0 = include videos in public data set
# 1 = do not include videos in public data set

public.permission <- experiment.data %>% 
    group_by(subject_id) %>% 
    filter(task == "public-videos") %>% 
    select("response") %>% 
    rename("public_perm" = "response")
  
metadata <- left_join(public.permission, metadata, by="subject_id")

research.permission <- experiment.data %>% 
    group_by(subject_id) %>% 
    filter(task == "research-videos") %>% 
    select("response") %>% 
    rename("research_perm" = "response")
  
# metadata <- left_join(research.permission, metadata, by="subject_id")

webm.files <- list.files(path='data/webm',
           pattern='.webm')

webms.per.subject <- data.frame(file_name = webm.files) %>% 
  mutate(subject_id = substring(file_name, 1, 8)) %>% 
  group_by(subject_id) %>% 
  count()

metadata <- left_join(webms.per.subject, metadata, by="subject_id")

detection.task.data <- json.data %>% 
  filter(task == "dot_detection")

# Record proportion of participant's reaction times above 200, 400, 800, 1600, and 3200 milliseconds

metadata <- detection.task.data %>% 
  select(rt, subject_id) %>% 
  group_by(subject_id) %>% 
  summarize(mean = mean(rt), median=median(rt), sd = sd(rt), 
            "prop<200ms" = sum(rt < 200) / n(), 
            "prop<400ms" = sum(rt < 400) / n(), 
            "prop<800ms" = sum(rt < 800) / n(),
            "prop<1600ms" = sum(rt < 1600) / n(),
            "prop<3200ms" = sum(rt < 3200) / n()) %>% 
  left_join(metadata, by="subject_id")

interacts.data <- NULL

# Concatenate json files into one megafile

# Creates a list of files from json directory
json_interact_files <- list.files(path='data/json',
                            pattern = "_interactions.json$",
                            full.names = TRUE)

for (file in json_interact_files) {
  subject = substring(file, 11, 18) # string with subject name
  df <- fromJSON(file) %>% 
    mutate(subject_id = subject)
  
  if(all(is.null(interacts.data))) {
    interacts.data <- df
  }
  else{
    interacts.data <- bind_rows(interacts.data, df)
  }

}

interacts.data.wide <- interacts.data %>% 
  group_by(subject_id) %>%
  filter(trial >= 9 & trial <= 340) %>% 
  count(event) %>%
  pivot_wider(id_cols=subject_id, names_from=event, values_from=n)
  
interacts.data.wide[is.na(interacts.data.wide)] <- 0
  
metadata <- left_join(interacts.data.wide, metadata, by="subject_id")

fullscreenexit.trial <- interacts.data %>% 
  filter(event == "fullscreenexit") %>% 
  group_by(subject_id) %>% 
  slice_head(n=1) %>% 
  select(subject_id, trial) %>% 
  rename("fsexit_trial" = "trial")

metadata <- left_join(fullscreenexit.trial, metadata, by="subject_id")




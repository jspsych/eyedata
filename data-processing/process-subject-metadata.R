setwd(here::here())

#### CONFIG ###########
retrieve_data <- TRUE
osf_access_token <- paste(
  readLines("data-processing/osf-token.txt"),
  collapse = "\n"
)
private_json_osf_component <- "vmdu6"
private_jpg_osf_component <- "3b6et"
private_webm_osf_component <- "34t8d"
private_osf_overall_component <- "2wcgp"

public_jpg_osf_component <- "7s2ae"
public_json_osf_component <- "6cn7a"
#######################

library(osfr)
library(tidyverse)
library(jsonlite)
# library(devtools)
# library(tidymedia)


if (retrieve_data) {
  if (!dir.exists("data")) {
    dir.create("data")
  }

  osf_auth(osf_access_token)

  osf_retrieve_node(private_json_osf_component) %>%
    osf_ls_files() %>%
    osf_download(path = "data", conflicts = "skip", progress = TRUE)
  osf_retrieve_node(private_jpg_osf_component) %>%
    osf_ls_files() %>%
    osf_download(path = "data", conflicts = "skip", progress = TRUE)


  untar("data/json.tar.gz", exdir = "data/json")
  untar("data/jpg.tar.gz", exdir = "data/jpg")
}

json_data <- list.files(
  path = "data/json",
  pattern = "^\\w{8}.json$",
  full.names = TRUE
) %>%
  lapply(fromJSON) %>%
  bind_rows()

prolific_info <- json_data %>%
  select(subject_id, prolific_id) %>%
  group_by(subject_id) %>%
  slice_head(n = 1)

data_trials <- c(
  "webcam-type",
  "glasses",
  "browser-check",
  "public-videos",
  "research-videos"
)

experiment_data <- json_data %>%
  filter((task %in% data_trials) | (trial_type == "html-video-response"))

people_who_give_permission_for_videos <- experiment_data %>%
  filter(task == "public-videos") %>%
  select(subject_id, response) %>% 
  filter(response == 0) %>%
  pull(subject_id)

jpg_files <- list.files(
  path = "data/jpg",
  pattern = ".jpg$",
)

jpg_files_per_subject <- data.frame(file_name = jpg_files) %>%
  mutate(subject_id = substring(file_name, 1, 8)) %>%
  group_by(subject_id) %>%
  count()

subjects_with_complete_data <- jpg_files_per_subject %>%
  filter(n == 144) %>%
  pull(subject_id)

subjects_with_complete_data_and_permission <- subjects_with_complete_data %>%
  intersect(people_who_give_permission_for_videos)

writeLines(toJSON(
  subjects_with_complete_data_and_permission
), "data/subjects-with-permission.json")

osf_retrieve_node(private_osf_overall_component) %>%
  osf_upload("data/subjects-with-permission.json", conflicts="overwrite")

webcam_info <- experiment_data %>%
  filter(task == "webcam-type") %>%
  select(subject_id, response) %>%
  mutate(
    webcam_type = factor(
      response,
      labels = c("integrated", "external"),
      levels = c(0, 1)
    )
  ) %>%
  select(-response)

glasses_info <- experiment_data %>%
  filter(task == "glasses") %>%
  select(subject_id, response) %>%
  mutate(
    wore_glasses = factor(
      response,
      labels = c("no", "yes"),
      levels = c(0, 1)
    )
  ) %>%
  select(-response)

screen_info <- experiment_data %>%
  filter(task == "browser-check") %>%
  select(subject_id, aspect_ratio, width, height)

detection_task_data <- json.data %>%
  filter(task == "dot_detection") %>%
  select(rt, subject_id) %>%
  group_by(subject_id) %>%
  summarize(
    mean = mean(rt), median = median(rt), sd = sd(rt),
    "prop<200ms" = sum(rt < 200) / n(),
    "prop<400ms" = sum(rt < 400) / n(),
    "prop<800ms" = sum(rt < 800) / n(),
    "prop<1600ms" = sum(rt < 1600) / n(),
    "prop<3200ms" = sum(rt < 3200) / n()
  )

interaction_data <- list.files(
  path = "data/json",
  pattern = "_interactions.json$",
  full.names = TRUE
) %>%
lapply(function(x){
  d <- fromJSON(x) %>%
  mutate(subject_id = substring(x, 11, 18))
  return(d)
}) %>%
bind_rows()

interaction_data_wide <- interaction_data %>%
  group_by(subject_id) %>%
  filter(trial >= 9 & trial <= 340) %>%
  count(event) %>%
  pivot_wider(id_cols = subject_id, names_from = event, values_from = n)

interaction_data_wide[is.na(interaction_data_wide)] <- 0

fullscreenexit_trial <- interaction_data %>%
  filter(event == "fullscreenexit") %>%
  group_by(subject_id) %>%
  slice_head(n = 1) %>%
  select(subject_id, trial) %>%
  rename("fullscreen_exit_trial" = "trial")

### create filtered jpg.tar.gz

public_jpg_files <- jpg_files[str_extract(jpg_files, "^[^_]+")  %in% subjects_with_complete_data_and_permission]

tar(
  tarfile = "data/public-jpg.tar.gz",
  files = file.path("data/jpg", public_jpg_files),
  compression = "gzip"
)

osf_retrieve_node(public_jpg_osf_component) %>%
  osf_upload("data/public-jpg.tar.gz", conflicts="overwrite")



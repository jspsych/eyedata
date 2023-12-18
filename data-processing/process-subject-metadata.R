setwd(here::here())

#### CONFIG
retrieve_data <- TRUE
osf_access_token <- paste(
  readLines("data-processing/osf-token.txt"),
  collapse = "\n"
)
private_json_osf_component <- "vmdu6"
private_jpg_osf_component <- "3b6et"
private_webm_osf_component <- "34t8d"

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

data_trials <- c(
  "webcam-type",
  "glasses",
  "browser-check",
  "public-videos",
  "research-videos"
)

experiment_data <- json_data %>%
  filter((task %in% data_trials) | (trial_type == "html-video-response"))

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

jpg_files <- list.files(
  path = "data/jpg",
  pattern = ".jpg$",
)

jpg_files_per_subject <- data.frame(file_name = jpg_files) %>%
  mutate(subject_id = substring(file_name, 1, 8)) %>%
  group_by(subject_id) %>%
  count()

metadata <- left_join(webms.per.subject, metadata, by = "subject_id")

detection_task_data <- json.data %>%
  filter(task == "dot_detection")

# Record proportion of participant's reaction times above 200, 400, 800, 1600, and 3200 milliseconds

metadata <- detection.task.data %>%
  select(rt, subject_id) %>%
  group_by(subject_id) %>%
  summarize(
    mean = mean(rt), median = median(rt), sd = sd(rt),
    "prop<200ms" = sum(rt < 200) / n(),
    "prop<400ms" = sum(rt < 400) / n(),
    "prop<800ms" = sum(rt < 800) / n(),
    "prop<1600ms" = sum(rt < 1600) / n(),
    "prop<3200ms" = sum(rt < 3200) / n()
  ) %>%
  left_join(metadata, by = "subject_id")

interacts.data <- NULL

# Concatenate json files into one megafile

# Creates a list of files from json directory
json_interact_files <- list.files(
  path = "data/json",
  pattern = "_interactions.json$",
  full.names = TRUE
)

for (file in json_interact_files) {
  subject <- substring(file, 11, 18) # string with subject name
  df <- fromJSON(file) %>%
    mutate(subject_id = subject)

  if (all(is.null(interacts.data))) {
    interacts.data <- df
  } else {
    interacts.data <- bind_rows(interacts.data, df)
  }
}

interacts.data.wide <- interacts.data %>%
  group_by(subject_id) %>%
  filter(trial >= 9 & trial <= 340) %>%
  count(event) %>%
  pivot_wider(id_cols = subject_id, names_from = event, values_from = n)

interacts.data.wide[is.na(interacts.data.wide)] <- 0

metadata <- left_join(interacts.data.wide, metadata, by = "subject_id")

fullscreenexit.trial <- interacts.data %>%
  filter(event == "fullscreenexit") %>%
  group_by(subject_id) %>%
  slice_head(n = 1) %>%
  select(subject_id, trial) %>%
  rename("fsexit_trial" = "trial")

metadata <- left_join(fullscreenexit.trial, metadata, by = "subject_id")

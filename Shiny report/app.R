library(shiny)
library(carData)
library(tidyverse)
theme_update(plot.title = element_text(hjust = 0.5))

data("SLID")
levels(SLID$language) <- c(levels(SLID$language), "Unknown")
SLID$language[is.na(SLID$language)] <- "Unknown"

ui <- fluidPage(
  navbarPage("Wages Analysis",
  tabPanel("EDA",
    titlePanel("Exploratory Data Analysis of Wages"),

    sidebarLayout(
        sidebarPanel(
          checkboxGroupInput("sex",
                        "Sex:",
                        choices = list(
                        "Male", "Female"
                        ),
                        selected = list("Male", "Female")),
          checkboxGroupInput("language",
                        "Language:",
                        choices = list(
                          "English", "French", "Other", "Unknown"
                        ),
                        selected = list("English", "French", "Other", "Unknown")),
          sliderInput("age",
                        "Age:",
                        min = min(SLID$age),
                        max = max(SLID$age),
                        value = c(min(SLID$age), max(SLID$age))),
          sliderInput("education",
                      "Years of education:",
                      min = min(SLID$education, na.rm = T),
                      max = max(SLID$education, na.rm = T),
                      value = c(min(SLID$education, na.rm = T), max(SLID$education, na.rm = T)))
        ),
        mainPanel(
            fluidRow(splitLayout(cellWidths = c("50%", "50%"), plotOutput("wage_dist"), plotOutput("wage_sex"))),
            fluidRow(splitLayout(cellWidths = c("50%", "50%"), plotOutput("wage_educ"), plotOutput("wage_age"))),
            fluidRow(splitLayout(cellWidths = c("50%", "50%"), plotOutput("wage_lang"), plotOutput("corr_plot")))
        )
    )
  ),
  tabPanel("Regression",
           
            titlePanel("Regression analysis of wages"),
           
            sidebarLayout(
                sidebarPanel(
                    textAreaInput("formula",
                        "Write a formula for explanatory variables:",
                        value = "education + age + sex + language"
                    ),
                    checkboxInput("log_trans",
                        "Should the wages be log transformed:",
                        F
                    ),
                    actionButton(
                        "calc", "calculate"
                    )
                ),
                mainPanel(
                  verbatimTextOutput("regression"), 
                  plotOutput("errors_hist"),
                  plotOutput("errors_fit")
                )
            )
        ),
  fluid = F)
)

server <- function(input, output) {
    filtered_data <- reactive({
        slid_filtered <- SLID %>%
        filter(language %in% input$language, age >= input$age[1], age <= input$age[2],
               sex %in% input$sex, education >= input$education[1], education <= input$education[2])
        return(slid_filtered)
    })
    model <- reactive({
        lm_model <- lm(as.formula(paste0(ifelse(input$log_trans,"log(wages)","wages"), " ~ ", input$formula)), data = SLID)
        return(lm_model)
    }) |> bindEvent(input$calc)
    output$regression <- renderPrint({
        summary(model())
    })
    output$errors_hist <- renderPlot({
        ggplot(mapping=aes(x=model()$residuals)) + geom_histogram() + ggtitle("Histogram of residuals") +
        xlab("residuals") + ylab("count")
    })
    output$errors_fit <- renderPlot({
        ggplot(mapping=aes(x=model()$fitted.values, y=model()$residuals)) + 
        geom_point() + ggtitle("Residuals vs fitted") + xlab("fitted values") + ylab("residuals")
    }) |> bindEvent(input$calc)
    output$wage_dist <- renderPlot({
        filtered_data() %>%
        ggplot(aes(x=wages)) + geom_histogram() + ggtitle("Hourly wages distribution") +
        xlab("hourly wage [USD]") + ylab("count")
    })
    output$wage_sex <- renderPlot({
        filtered_data() %>%
        group_by(sex) %>%
        summarise("wage_avg" = mean(wages, na.rm = T)) %>%
        ggplot(aes(x=sex, y=wage_avg, fill=sex)) + geom_bar(stat = "identity", show.legend = F) + 
        ggtitle("Average hourly wage by sex") + xlab("sex") + ylab("avearge hourly wage [USD]") +
        scale_fill_manual(values = c("#F084C6", "#17CAEF"))
    })
    output$wage_educ <- renderPlot({
        filtered_data() %>%
        ggplot(aes(x=wages, y=education)) + geom_point() + ggtitle("Hourly wages vs years of education") +
        xlab("hourly wage [USD]") + ylab("years of education")
    })
    output$wage_age <- renderPlot({
        filtered_data() %>%
        ggplot(aes(x=wages, y=age)) + geom_point() + ggtitle("Hourly wages vs age") +
        xlab("hourly wage [USD]") + ylab("age")
    })
    output$wage_lang <- renderPlot({
        filtered_data() %>%
        group_by(language) %>%
        summarise("wage_avg" = mean(wages, na.rm = T)) %>%
        ggplot(aes(x=language, y=wage_avg, fill=language)) + geom_bar(stat = "identity", show.legend = F) + 
        ggtitle("Average hourly wage by language") + xlab("language") + ylab("avearge hourly wage [USD]") +
        scale_fill_manual(values = c("#CE1B09", "#060270", "#06EC2C", "#11D4EE"))
    })
    output$corr_plot <- renderPlot({
        corrplot::corrplot(
          cor(
            filtered_data() %>%
            select(wages, education, age),
            use = "complete.obs"
          ),
          "number"
        )
    })
}

shinyApp(ui = ui, server = server)

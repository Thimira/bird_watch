var $ratingStars;

$(function() {
    $ratingStars = $('.rate-star');

    $ratingStars.on('click', function () {
        var index = $(this).attr('data-index');
        markStarsAsActive(index);
    });

    function markStarsAsActive(index) {
        unmarkActive();
        for (var i = 0; i <= index; i++) {
            $($ratingStars.get(i)).addClass('amber-text');
        }
        $("#star-value").html(parseInt(index)+1);
    }

    function unmarkActive() {
        $ratingStars.removeClass('amber-text');
    }

    $("button#submit-feedback").click( function(event) {
        var rating = $("#star-value").text().trim();
        var feedback = $("#feedback-text").val().trim();

        if ((rating && rating.length !== 0) && (feedback && feedback.length !== 0)) {
            $.ajax({
                type: "POST",
                url: "/feedback",
                data: JSON.stringify({ 
                    rating: rating,
                    feedback: feedback
                }),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
                success: function(data) {
                    var message = getFeedbackMessage("Thank you! Your feedback has been recorded.", "success");
                    $("#feedback-ack").html(message);
                },
                failure: function(errMsg) {
                    // alert(errMsg);
                }
            });
        } else {
            var message = getFeedbackMessage("Please add a rating and a feedback message.", "warning");
            $("#feedback-ack").html(message);               
        }
    });

    function getFeedbackMessage(displayMessage, alertType) {
        var message = '<div class="alert alert-' + alertType + ' alert-dismissible fade show" role="alert">'
                        + displayMessage
                        + '<button type="button" class="close" data-dismiss="alert" aria-label="Close">'
                        +     '<span aria-hidden="true">&times;</span>'
                        + '</button>'
                    + '</div>';
        return message;
    }

});


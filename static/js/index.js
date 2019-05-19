$(function() {
    function setCorrectness(correctness) {
        var prediction_id = $("#prediction_id").html();
        $.ajax({
            type: "POST",
            url: "/correctness",
            data: JSON.stringify({ 
                prediction_id: prediction_id,
                correctness: correctness
            }),
            contentType: "application/json; charset=utf-8",
            dataType: "json",
            success: function(data) {
                $("#correctness-ack").show();
            },
            failure: function(errMsg) {
                // alert(errMsg);
            }
        });
    }

    $("#pred-correct").click( function(event) {
        event.preventDefault();
        $("#pred-correct").removeClass("badge-secondary");
        $("#pred-correct").addClass("badge-success");
        $("#pred-incorrect").removeClass("badge-danger");
        $("#pred-incorrect").addClass("badge-secondary");
        $("#pred-notsure").removeClass("badge-info");
        $("#pred-notsure").addClass("badge-secondary");
        setCorrectness(1);

    });

    $("#pred-incorrect").click( function(event) {
        event.preventDefault();
        $("#pred-correct").removeClass("badge-success");
        $("#pred-correct").addClass("badge-secondary");
        $("#pred-incorrect").removeClass("badge-secondary");
        $("#pred-incorrect").addClass("badge-danger");
        $("#pred-notsure").removeClass("badge-info");
        $("#pred-notsure").addClass("badge-secondary");
        setCorrectness(0);
    });

    $("#pred-notsure").click( function(event) {
        event.preventDefault();
        $("#pred-correct").removeClass("badge-success");
        $("#pred-correct").addClass("badge-secondary");
        $("#pred-incorrect").removeClass("badge-danger");
        $("#pred-incorrect").addClass("badge-secondary");
        $("#pred-notsure").removeClass("badge-secondary");
        $("#pred-notsure").addClass("badge-info");
        setCorrectness(-1);
    });
});